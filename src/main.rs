use libopus::decoder::*;
use libopus::encoder::*;

use std::io::{self, Error, Read, Write};
use structopt::StructOpt;

use wavey::av::{frame_spectrogram, save_spectrogram};
use wavey::packet::{decode_audio_packet_header, encode_audio_packet, HEADER_SIZE};
use wavey::types::{AudioConfig, EncodingFlag};
use wavey::utils::deinterleave_vecs_f32;

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
            const WIN: usize = 768;
            const HOP: usize = 256;
            const NORM: f32 = 0.0;

            let len = bytes_per_sample as usize * channel_count as usize * WIN;

            let mut buffer = vec![0; len];
            let mut channel_frames: Vec<Vec<Vec<f32>>> = vec![Vec::new(); channel_count as usize];

            loop {
                let bytes_read = input.read(&mut buffer)?;
                accumulated_bytes.extend_from_slice(&buffer[..bytes_read]);
                if bytes_read == 0 || accumulated_bytes.len() >= len {
                    while accumulated_bytes.len() >= len {
                        let (chunk, rest) = accumulated_bytes.split_at_mut(len);
                        let data: Vec<Vec<f32>> =
                            deinterleave_vecs_f32(&chunk, channel_count as usize)
                                .into_iter()
                                .map(|mut d| {
                                    if d.len() % WIN != 0 {
                                        d.extend(vec![0.0; WIN - d.len()]);
                                    }
                                    d
                                })
                                .collect();
                        for (i, pcm_data) in data.into_iter().enumerate() {
                            let frame_data = frame_spectrogram(pcm_data, WIN, HOP, NORM);
                            channel_frames[i].extend(frame_data);
                        }

                        accumulated_bytes = rest.to_vec();
                    }
                }

                if bytes_read == 0 {
                    break;
                }
            }
            if channel_frames[0].len() > 0 {
                save_spectrogram(&channel_frames, true, "out/sono-eq", 48000);
                save_spectrogram(&channel_frames, false, "out/sono-ue", 48000);
            }

            Ok(())
        }
        None => {
            eprintln!("No command specified");
            std::process::exit(1);
        }
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
