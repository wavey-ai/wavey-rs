use libopus::decoder::*;
use libopus::encoder::*;

use std::io::{self, Error, Read, Write};
use structopt::StructOpt;

use wavey::av::{
    process_sonograph_frame, process_waveform_frame, save_spectrogram, u16vec_to_bytes,
};

use wavey::packet::{decode_audio_packet_header, encode_audio_packet, HEADER_SIZE};
use wavey::types::{AudioConfig, EncodingFlag};
use wavey::utils::{deinterleave_vecs_f32, interleave_vecs_u8};

use image::ImageBuffer;
use image::Rgba;
use rustfft::{Fft, FftPlanner};
use std::collections::HashMap;
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

            const WIN: usize = 768;
            const HOP: usize = 256;

            let (sender, receiver) = mpsc::channel();
            let process_thread = thread::spawn(move || {
                let receiver = Arc::new(Mutex::new(receiver));
                process_tasks(
                    receiver,
                    WIN,
                    HOP,
                    args.samples_per_pixel,
                    channel_count,
                    args.workers,
                );
            });

            let mut channel_frames: Vec<Vec<Vec<f32>>> = vec![Vec::new(); channel_count as usize];
            let mut channel_waveforms: Vec<Vec<u8>> = vec![Vec::new(); channel_count as usize];
            let mut frame_count: u32 = 0;

            let bytes_per_samples = bytes_per_sample as usize * channel_count as usize;
            let BUFFER_SIZE: usize = (bytes_per_samples * WIN) + (HOP * 2);
            let mut buffer = vec![0; BUFFER_SIZE];
            loop {
                let bytes_read = input.read(&mut buffer)?;
                if bytes_read == 0 {
                    break;
                }

                accumulated_bytes.extend_from_slice(&buffer[..bytes_read]);

                if accumulated_bytes.len() >= BUFFER_SIZE {
                    while accumulated_bytes.len() >= BUFFER_SIZE {
                        let (chunk, rest) = accumulated_bytes.split_at(BUFFER_SIZE);
                        sender.send((frame_count as usize, chunk.to_vec())).unwrap();
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

                sender
                    .send((frame_count as usize, accumulated_bytes.to_vec()))
                    .unwrap();
                frame_count += 1;
            }

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
    receiver: Arc<Mutex<Receiver<(usize, Vec<u8>)>>>,
    win: usize,
    hop: usize,
    samples_per_point: usize,
    channel_count: u8,
    num_workers: usize,
) {
    let mut handles = Vec::new();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(win);

    for _ in 0..num_workers {
        let receiver = Arc::clone(&receiver);
        let fft_clone = Arc::clone(&fft);

        let handle = thread::spawn(move || {
            let hamming_window: Vec<f32> = (0..win)
                .map(|i| {
                    0.54 - 0.46
                        * ((2.0 * std::f32::consts::PI * i as f32) / (win as f32 - 1.0)).cos()
                })
                .collect();

            let mut waveforms: HashMap<usize, Vec<u8>> = HashMap::new();

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

                let (chunk_idx, chunk) = task.unwrap();
                let data: Vec<Vec<f32>> = deinterleave_vecs_f32(&chunk, channel_count as usize);
                let mut channel_waveforms: Vec<Vec<u8>> = vec![Vec::new(); channel_count as usize];
                for (i, pcm_data) in data.into_iter().enumerate() {
                    let waveform_segment = process_waveform_frame(
                        &pcm_data,
                        win,
                        hop,
                        samples_per_point,
                        hamming_window.clone(),
                    );
                    channel_waveforms[i] = waveform_segment;
                    let sonograph_segment = process_sonograph_frame(
                        &pcm_data,
                        win,
                        hop,
                        hamming_window.clone(),
                        &fft_clone,
                    );
                }

                let interleaved_bytes = interleave_vecs_u8(&channel_waveforms);
                let waveform = waveforms.entry(chunk_idx).or_insert(interleaved_bytes);
            }

            waveforms
        });

        handles.push(handle);
    }

    let mut merged_waveforms: HashMap<usize, Vec<u8>> = HashMap::new();

    for handle in handles {
        let waveform = handle.join().unwrap();
        for (key, value) in waveform {
            merged_waveforms.entry(key).or_insert(value);
        }
    }

    let mut sorted_waveforms: Vec<(usize, Vec<u8>)> = merged_waveforms.into_iter().collect();
    sorted_waveforms.sort_by_key(|&(key, _)| key);
    let merged_values: Vec<u8> = sorted_waveforms
        .into_iter()
        .flat_map(|(_, values)| values)
        .collect();

    let mut header = vec![channel_count];
    header.extend_from_slice(&merged_values);
    let name = format!("{}/{}", "out", "waveform.raw");
    let path = Path::new(&name);

    std::fs::write(path, &header).unwrap();
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
        AudioConfig::Hz48000Bit32,
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
