use libopus::decoder::*;
use libopus::encoder::*;
use std::io::{BufRead, BufReader, Read};

use crate::types::*;
use crate::utils::*;

pub const HEADER_SIZE: usize = 4;

pub fn encode_audio_packet(
    inputFormat: AudioConfig,
    buf: Vec<u8>,
    channel_count: usize,
    bits_per_sample: u8,
    format: &EncodingFlag,
    encoder: &mut Encoder,
    max_payload_size: usize,
) -> Result<Vec<Vec<u8>>, String> {
    let (sampling_rate, src_bits_per_sample) = get_sampling_rate_and_bits_per_sample(inputFormat);

    let config = match get_audio_config(sampling_rate, bits_per_sample) {
        Ok(c) => c,
        Err(err) => return Err(err),
    };

    let bytes_per_sample = bits_per_sample / 8;
    let frame_size = buf.len() as u32 / bytes_per_sample as u32;
    let len = frame_size as usize * bytes_per_sample as usize;

    let mut data = vec![0u8; len as usize];

    match &format {
        EncodingFlag::Opus => {
            let mut src: Vec<i16> = Vec::new();
            match src_bits_per_sample {
                16 => {
                    for bytes in buf.chunks_exact(2) {
                        let sample = i16::from_le_bytes([bytes[0], bytes[1]]);
                        src.push(sample);
                    }
                }
                32 => {
                    for bytes in buf.chunks_exact(4) {
                        let sample = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                        let scaled_sample = (sample * 32767.0) as i16;
                        src.push(scaled_sample);
                    }
                }
                _ => {
                    return Err(format!(
                        "Unsupported bits per_sample: {}",
                        src_bits_per_sample
                    ))
                }
            }

            let mut dst = vec![0u8; frame_size as usize * 2];
            let mut num_bytes = encoder.encode(&src[..], &mut data[..]).unwrap_or(0);
            if num_bytes == 0 {
                return Err("opus enc: got zero bytes".to_string());
            }
            data.truncate(num_bytes);
        }
        EncodingFlag::PCM => match bits_per_sample {
            16 => match src_bits_per_sample {
                16 => {
                    data.clone_from_slice(&buf[..]);
                }
                32 => {
                    for bytes in buf.chunks_exact(4) {
                        let sample = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                        let scaled_sample = (sample * 32767.0) as i16;
                        let bytes = scaled_sample.to_le_bytes();
                        data.extend_from_slice(&bytes);
                    }
                }
                _ => {
                    return Err(format!(
                        "Unsupported src bits per_sample: {}",
                        src_bits_per_sample
                    ))
                }
            },
            32 => match src_bits_per_sample {
                32 => {
                    data.clone_from_slice(&buf[..]);
                }
                _ => {
                    return Err(format!(
                        "Unsupported source bits per_sample: {}",
                        src_bits_per_sample
                    ))
                }
            },
            _ => return Err(format!("Unsupported bits per_sample: {}", bits_per_sample)),
        },
    };

    let mut result_chunks = Vec::new();

    if max_payload_size == 0 || len <= max_payload_size - HEADER_SIZE {
        let mut chunk = Vec::new();
        let header = encode_audio_packet_header(
            format,
            config.clone(),
            channel_count as u8,
            data.len() as u16,
        );
        chunk.extend_from_slice(&header);
        chunk.extend_from_slice(&data);

        result_chunks.push(chunk);
    } else {
        let max_audio_payload = max_payload_size - HEADER_SIZE;
        let total_size = len;
        let chunk_count = (total_size + max_audio_payload - 1) / max_audio_payload;

        for chunk_index in 0..chunk_count {
            let mut chunk_data = Vec::new();
            let mut frame_size = 0;

            let start = chunk_index * max_audio_payload / channel_count;
            let end = start + max_audio_payload.min(chunk_data.len() - start);
            chunk_data.extend_from_slice(&data[start..end]);
            frame_size = end - start;

            let header = encode_audio_packet_header(
                format,
                config.clone(),
                channel_count as u8,
                frame_size as u16,
            );

            let mut chunk = Vec::new();
            chunk.extend_from_slice(&header);
            chunk.extend_from_slice(&chunk_data);

            result_chunks.push(chunk);
        }
    }

    Ok(result_chunks)
}

pub fn decode_audio_packet(buffer: Vec<u8>, decoder: &mut Decoder) -> Option<AudioList> {
    let header = decode_audio_packet_header(&buffer);
    let channel_count = header.channel_count as usize;
    let encoding_flag = header.encoding;
    let bytes_per_sample = header.bits_per_sample as usize / 8;
    let len = buffer.len() - HEADER_SIZE;
    let mut sample_count: usize = len / (channel_count * bytes_per_sample);
    let data = &buffer[HEADER_SIZE..];
    let mut samples = vec![0.0f32; sample_count * channel_count];

    match encoding_flag {
        EncodingFlag::PCM => {
            if bytes_per_sample == 2 {
                for sample_bytes in data.chunks_exact(2) {
                    let sample_i16 = i16::from_le_bytes([sample_bytes[0], sample_bytes[1]]);
                    let sample_f32 = f32::from(sample_i16) / f32::from(std::i16::MAX);
                    samples.push(sample_f32);
                }
            } else if bytes_per_sample == 3 {
                for sample_bytes in data.chunks_exact(3) {
                    let sample_i24 =
                        i32::from_le_bytes([sample_bytes[0], sample_bytes[1], sample_bytes[2], 0]);
                    let sample_f32 = sample_i24 as f32 / (1 << 23) as f32;
                    samples.push(sample_f32);
                }
            } else if bytes_per_sample == 4 {
                for sample_bytes in data.chunks_exact(4) {
                    let sample_f32 = f32::from_le_bytes([
                        sample_bytes[0],
                        sample_bytes[1],
                        sample_bytes[2],
                        sample_bytes[3],
                    ]);
                    samples.push(sample_f32);
                }
            }
        }
        EncodingFlag::Opus => {
            let mut dst = vec![0i16; sample_count * channel_count];
            let num_samples = decoder.decode(&data[..], &mut dst[..], false).unwrap_or(0);

            for sample_i16 in dst {
                let sample_f32 = f32::from(sample_i16) / f32::from(std::i16::MAX);
                samples.push(sample_f32);
            }
        }
        _ => return None,
    };

    let mut deinterleaved_samples: Vec<Vec<f32>> =
        vec![Vec::with_capacity(samples.len() / channel_count); channel_count];
    for (i, sample) in samples.iter().enumerate() {
        deinterleaved_samples[i % channel_count].push(*sample);
    }

    Some(AudioList {
        channels: deinterleaved_samples,
        sampling_rate: header.sampling_rate as usize,
        sample_count: sample_count,
    })
}

fn encode_audio_packet_header(
    encoding: &EncodingFlag,
    config: AudioConfig,
    channel_count: u8,
    frame_size: u16,
) -> Vec<u8> {
    let mut flag: u8 = 0;
    if encoding == &EncodingFlag::Opus {
        flag = 1;
    };

    let mut id = config as u8;
    id |= flag << 5;
    let mut data = id.to_le_bytes().to_vec();
    data.push(channel_count);
    data.extend_from_slice(&frame_size.to_le_bytes());
    data
}

pub fn decode_audio_packet_header(data: &Vec<u8>) -> AudioPacketHeader {
    let flag = if (data[0] & 0x20) == 0 {
        EncodingFlag::PCM
    } else {
        EncodingFlag::Opus
    };
    let config_id = data[0] & 0x1F; // Extract the  config ID from the first byte of the header
    let config = get_config(config_id);
    let channel_count = data[1];
    let frame_size = u16::from_le_bytes([data[2], data[3]]);
    let (sampling_rate, bits_per_sample) = get_sampling_rate_and_bits_per_sample(config);
    return AudioPacketHeader {
        encoding: flag,
        sampling_rate: sampling_rate,
        bits_per_sample: bits_per_sample,
        channel_count: channel_count,
        frame_size: frame_size,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Read;

    fn read_test_file(filename: &str) -> Vec<u8> {
        let mut file = File::open(filename).unwrap();
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();
        buffer
    }

    #[test]
    fn test_decode_audio_packet_header() {
        let encoded =
            encode_audio_packet_header(&EncodingFlag::Opus, AudioConfig::Hz44100Bit32, 2, 400u16);
        let got = decode_audio_packet_header(&encoded);

        assert_eq!(got.encoding, EncodingFlag::Opus);
        assert_eq!(got.sampling_rate, 44100);
        assert_eq!(got.bits_per_sample, 32);
        assert_eq!(got.channel_count, 2);
        assert_eq!(got.frame_size, 400u16);
    }

    #[test]
    fn test_encode_audio_packet_header_pcm() {
        let got =
            encode_audio_packet_header(&EncodingFlag::PCM, AudioConfig::Hz44100Bit32, 2, 400u16);
        let flag = if (got[0] & 0x20) == 0 {
            EncodingFlag::PCM
        } else {
            EncodingFlag::Opus
        };
        assert_eq!(flag, EncodingFlag::PCM);
    }

    #[test]
    fn test_encode_audio_packet_header_opus() {
        let got =
            encode_audio_packet_header(&EncodingFlag::Opus, AudioConfig::Hz44100Bit32, 2, 400u16);
        let flag = if (got[0] & 0x20) == 0 {
            EncodingFlag::PCM
        } else {
            EncodingFlag::Opus
        };
        assert_eq!(flag, EncodingFlag::Opus);
    }

    #[test]
    fn test_encode_audio_packet_header() {
        let got =
            encode_audio_packet_header(&EncodingFlag::Opus, AudioConfig::Hz44100Bit32, 2, 400u16);
        let flag = if (got[0] & 0x20) == 0 {
            EncodingFlag::PCM
        } else {
            EncodingFlag::Opus
        };
        let config_id = got[0] & 0x1F; // Extract the audio config ID from the first byte of the header
        let decoded_config = match config_id {
            2 => AudioConfig::Hz44100Bit32,
            _ => panic!("Invalid config ID"),
        };
        assert_eq!(got.len() * std::mem::size_of::<u8>(), HEADER_SIZE);
        assert_eq!(flag, EncodingFlag::Opus);
        assert_eq!(decoded_config, AudioConfig::Hz44100Bit32);
        assert_eq!(got[2..4], 400u16.to_le_bytes(),);
    }

    fn test_encode_decode_common(bits_per_sample: u32) {
        let buffer = read_test_file("./test/f32.wav");
        let pcm_data = buffer[44..].to_vec();
        let channel_count = 2;
        let sampling_rate = 48000;
        let frame_count = pcm_data.len() as u32 / (channel_count * bits_per_sample / 8) as u32;
        let mut encoder = Encoder::create(48000, 2, 1, 1, &[0u8, 1u8], Application::Audio).unwrap();

        let encoded_buffers = encode_audio_packet(
            AudioConfig::Hz44100Bit32,
            pcm_data,
            channel_count as usize,
            bits_per_sample as u8,
            &EncodingFlag::PCM,
            &mut encoder,
            0,
        )
        .unwrap();

        let mut decoded_samples_total = Vec::new();
        let mut decoder = Decoder::create(48000, 2, 1, 1, &[0u8, 1u8]).unwrap();

        for encoded_buffer in encoded_buffers {
            let decoded_samples = decode_audio_packet(encoded_buffer, &mut decoder).unwrap();
            decoded_samples_total.push(decoded_samples);
        }
    }

    #[test]
    fn test_encode_decode_pcm32() {
        test_encode_decode_common(32);
    }

    #[test]
    fn test_encode_decode_pcm16() {
        test_encode_decode_common(16);
    }
}
