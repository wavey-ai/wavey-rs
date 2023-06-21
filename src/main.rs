use libopus::decoder::*;
use libopus::encoder::*;

use std::io::{self, Error, Read, Write};
use structopt::StructOpt;

use wavey::av::{process_frame, save_spectrogram, u16vec_to_bytes};

use wavey::packet::{decode_audio_packet_header, encode_audio_packet, HEADER_SIZE};
use wavey::types::{AudioConfig, EncodingFlag};
use wavey::utils::deinterleave_vecs_f32;

use image::ImageBuffer;
use image::Rgba;
use std::path::Path;
use std::sync::mpsc::{self, Receiver};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

#[derive(Debug, StructOpt)]
#[structopt(name = "soundkit", about = "An audio encoding utility.")]
struct Command {
    #[structopt(subcommand)]
    cmd: Option<SubCommand>,
    #[structopt(short, long, default_value = "48000")]
    sampling_rate: u32,
    #[structopt(short, long, default_value = "32")]
    bits_per_sample: u8,
    #[structopt(short, long, default_value = "2")]
    channel_count: u8,
    #[structopt(short, long, default_value = "out")]
    out_dir: String,
    #[structopt(short, long, default_value = "4")]
    workers: usize,
    #[structopt(long, default_value = "12")]
    samples_per_pixel: usize,
}
#[derive(Debug, StructOpt)]
enum SubCommand {
    Encode,
    Images,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut input: Box<dyn Read> = Box::new(io::stdin());
    let mut output: Box<dyn Write> = Box::new(io::stdout());
    let mut accumulated_bytes = Vec::new();

    let args = Command::from_args();

    let frame_size: u16 = 120;
    let sampling_rate = args.sampling_rate;
    let bits_per_sample = args.bits_per_sample;
    let channel_count = args.channel_count;
    let bytes_per_sample = bits_per_sample / 8;
    println!("Sampling Rate: {}", args.sampling_rate);
    println!("Bits per Sample: {}", args.bits_per_sample);
    println!("Channel Count: {}", args.channel_count);
    match args.cmd {
        Some(SubCommand::Encode) => {
            let mut encoded_data = Vec::new();
            let mut offset: u32 = 0;
            let mut offsets = Vec::new();

            let len = bytes_per_sample as usize * channel_count as usize * frame_size as usize;

            let mut buffer = vec![0; len];

            let mut opus_encoder =
                Encoder::create(48000, 2, 1, 1, &[0u8, 1u8], Application::Audio).unwrap();
            opus_encoder
                .set_option(OPUS_SET_BITRATE_REQUEST, 96000)
                .unwrap();

            loop {
                let bytes_read = input.read(&mut buffer)?;
                accumulated_bytes.extend_from_slice(&buffer[..bytes_read]);

                while accumulated_bytes.len() >= len {
                    let (chunk, rest) = accumulated_bytes.split_at_mut(len);
                    let flag = EncodingFlag::Opus;
                    process_audio_chunk(
                        chunk.to_vec(),
                        channel_count,
                        bits_per_sample,
                        &flag,
                        &mut opus_encoder,
                        &mut offset,
                        &mut offsets,
                        &mut encoded_data,
                    )?;
                    accumulated_bytes = rest.to_vec();
                }

                if bytes_read == 0 {
                    let flag = EncodingFlag::PCM;
                    process_audio_chunk(
                        accumulated_bytes,
                        channel_count,
                        bits_per_sample,
                        &flag,
                        &mut opus_encoder,
                        &mut offset,
                        &mut offsets,
                        &mut encoded_data,
                    )?;

                    break;
                }
            }
            let mut final_encoded_data = Vec::new();
            for i in 0..4 {
                final_encoded_data.push(((offsets.len() >> (i * 8)) & 0xFF) as u8);
            }

            for offset in offsets {
                for i in 0..4 {
                    final_encoded_data.push((offset >> (i * 8) & 0xFF) as u8);
                }
            }

            final_encoded_data.extend(encoded_data);

            output.write_all(&final_encoded_data)?;

            Ok(())
        }
        Some(SubCommand::Images) => {
            let start = Instant::now();

            let (sender, receiver) = mpsc::channel();
            let out_dir = args.out_dir.clone();

            let process_thread = thread::spawn(move || {
                let receiver = Arc::new(Mutex::new(receiver));
                process_tasks(receiver, out_dir.clone(), args.workers);
            });

            let mut channel_frames: Vec<Vec<Vec<f32>>> = vec![Vec::new(); channel_count as usize];
            let spp = args.samples_per_pixel;
            let mut frame_count: u16 = 0;

            const WIN: usize = 768;
            const HOP: usize = 256;

            let bytes_per_samples = bytes_per_sample as usize * channel_count as usize;
            let BUFFER_SIZE: usize = bytes_per_samples * (WIN * 8);
            let mut buffer = vec![0; BUFFER_SIZE];
            let mut waveform: Vec<u16> = Vec::new();
            loop {
                let bytes_read = input.read(&mut buffer)?;
                if bytes_read == 0 {
                    break;
                }

                accumulated_bytes.extend_from_slice(&buffer[..bytes_read]);

                if accumulated_bytes.len() >= BUFFER_SIZE {
                    while accumulated_bytes.len() >= BUFFER_SIZE {
                        let (chunk, rest) = accumulated_bytes.split_at_mut(BUFFER_SIZE);
                        let data: Vec<Vec<f32>> =
                            deinterleave_vecs_f32(&chunk, channel_count as usize);
                        for (i, pcm_data) in data.clone().into_iter().enumerate() {
                            let (spectrogram_frames, waveform_segment) =
                                process_frame(pcm_data.clone(), WIN, HOP, spp);
                            channel_frames[i].extend(spectrogram_frames);
                            waveform.extend_from_slice(&waveform_segment);
                        }
                        frame_count += 1;

                        accumulated_bytes = rest.to_vec();
                    }
                }
            }

            if !accumulated_bytes.is_empty() {
                let data: Vec<Vec<f32>> =
                    deinterleave_vecs_f32(&accumulated_bytes, channel_count as usize)
                        .into_iter()
                        .map(|mut d| {
                            if d.len() % WIN != 0 {
                                let additional_samples = WIN - (d.len() % WIN);
                                d.extend(vec![0.0; additional_samples]);
                            }
                            d
                        })
                        .collect();
                for (i, pcm_data) in data.into_iter().enumerate() {
                    let (spectrogram_frames, waveform_segment) =
                        process_frame(pcm_data, WIN, HOP, spp);
                    channel_frames[i].extend(spectrogram_frames);
                    waveform.extend_from_slice(&waveform_segment);
                }
                frame_count += 1;
            }

            /*         save_spectrogram(
                            &channel_frames,
                            true,
                            &args.out_dir,
                            "sono-eq",
                            sampling_rat,
                        );
                        save_spectrogram(
                            &channel_frames,
                            false,
                            &args.out_dir,
                            "sono-ue",
                            sampling_rate,
                        );
            */

            let mut header = (WIN as u16).to_le_bytes().to_vec();
            header.extend_from_slice(&(spp as u16).to_le_bytes());
            header.extend_from_slice(&(channel_count as u16).to_le_bytes());
            header.extend_from_slice(&(frame_count as u16).to_le_bytes());
            header.extend_from_slice(&u16vec_to_bytes(&waveform));

            let name = format!("{}/{}", args.out_dir, "waveform.raw");
            let path = Path::new(&name);

            std::fs::write(path, &header).unwrap();

            drop(sender); // Signal the end of tasks

            process_thread.join().unwrap();

            let duration = start.elapsed();
            println!("Execution time: {:?}", duration);

            Ok(())
        }
        None => {
            eprintln!("No command specified");
            std::process::exit(1);
        }
    }
}

fn process_tasks(
    receiver: Arc<Mutex<Receiver<(usize, ImageBuffer<Rgba<u8>, Vec<u8>>)>>>,
    out_dir: String,
    num_workers: usize,
) {
    let mut handles = Vec::new();

    for _ in 0..num_workers {
        let receiver = Arc::clone(&receiver);
        let out_dir_clone = out_dir.clone();

        let handle = thread::spawn(move || {
            loop {
                // Acquire the lock on the receiver
                let task = {
                    let lock = receiver.lock().unwrap();
                    lock.recv().ok()
                };

                // Check if the channel is closed or no more tasks are available
                if task.is_none() {
                    break;
                }

                let (chunk_idx, imgbuf) = task.unwrap();

                let name = format!("{}/{}-{}.webp", out_dir_clone, "wave", chunk_idx);
                let path = Path::new(&name);
                dbg!(path);
                imgbuf.save(path).unwrap();
            }
        });

        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

fn process_audio_chunk(
    chunk: Vec<u8>,
    channel_count: u8,
    bits_per_sample: u8,
    flag: &EncodingFlag,
    opus_encoder: &mut Encoder,
    offset: &mut u32,
    offsets: &mut Vec<u32>,
    encoded_data: &mut Vec<u8>,
) -> Result<(), Box<dyn std::error::Error>> {
    let packets = encode_audio_packet(
        AudioConfig::Hz48000Bit16,
        chunk,
        channel_count as usize,
        bits_per_sample as u8,
        flag,
        opus_encoder,
        0,
    )?;

    for packet_chunk in packets {
        let header = decode_audio_packet_header(&packet_chunk);
        offsets.push(*offset);
        *offset += HEADER_SIZE as u32 + header.frame_size as u32;
        encoded_data.extend(packet_chunk);
    }

    Ok(())
}
