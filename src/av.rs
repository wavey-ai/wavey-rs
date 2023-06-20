use image::{DynamicImage, GrayImage, ImageBuffer, Luma, Rgba, RgbaImage};
use imageproc::contrast::equalize_histogram;
use rustfft::algorithm::Radix4;
use rustfft::num_traits::Zero;
use rustfft::{num_complex::Complex, FftPlanner};
use rusttype::{Font, Scale};
use std::cmp;
use std::io::Cursor;
use std::path::Path;

use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread;

pub fn waveform_image(data: &[Vec<Vec<(u32, u32, Rgba<u8>)>>], height: u32) -> RgbaImage {
    let mut max_x = 0;
    let mut offset = 0;

    for buffer_data in data {
        let mut x_offset = 0;
        for channel_data in buffer_data {
            for &(x, _, _) in channel_data {
                if x > x_offset {
                    x_offset = x;
                }
            }
        }
        offset += x_offset;
    }

    let mut imgbuf = ImageBuffer::new(offset + 1, height);

    offset = 0;

    for buffer_data in data {
        let mut x_offset = 0;
        for channel_data in buffer_data {
            for &(x, y, color) in channel_data {
                imgbuf.put_pixel(x + offset, y, color);
                if x > x_offset {
                    x_offset = x;
                }
            }
        }
        offset += x_offset;
    }

    imgbuf
}

pub fn save_spectrogram(
    frames: &Vec<Vec<Vec<f32>>>,
    eq: bool,
    out_dir: &str,
    fileprefix: &str,
    sample_rate: u32,
) {
    let width = frames[0].len();
    let height = frames[0][0].len();

    for i in 0..frames.len() {
        let tile_width = 10000;
        let mut start_pixel = 0;

        while start_pixel < width {
            // Calculate the end pixel of the current tile
            let end_pixel = (start_pixel + tile_width).min(width);
            let tile_width_current = end_pixel - start_pixel;

            let mut imgbuf = GrayImage::new(tile_width_current as u32, height as u32);

            for (x, frame) in frames[i][start_pixel..end_pixel].iter().enumerate() {
                let frame: Vec<f32> = frame.iter().rev().copied().collect();
                for (y, &value) in frame.iter().enumerate() {
                    let pixel = ((value * 255.0) as u8).min(255).max(0);
                    imgbuf.put_pixel(x as u32, y as u32, Luma([pixel]));
                }
            }

            let name: String = format!("{}/{}{}_{}.png", out_dir, fileprefix, i, start_pixel);
            imgbuf.save(name).unwrap();
            start_pixel += tile_width;
        }
    }
}

pub fn data_to_bytes(data: &[(u16, u16)]) -> Vec<u8> {
    let total_elements = data.len();
    let buffer_size = total_elements * std::mem::size_of::<u16>();
    let mut buf: Vec<u8> = Vec::with_capacity(buffer_size);

    for &(x, y) in data {
        let x_bytes = u16::to_le_bytes(x);
        let y_bytes = u16::to_le_bytes(y);

        buf.extend_from_slice(&x_bytes);
        buf.extend_from_slice(&y_bytes);
    }

    buf
}

pub fn waveform_vector_data(data: Vec<f32>, samples_per_pixel: usize, height: usize) -> Vec<(u16, u16)> {
    let mut waveform: Vec<(u16, u16)> = Vec::new();

    for chunk in data.chunks(samples_per_pixel) {
        let (positive_avg, negative_avg) = average_values(chunk);
        waveform.push(
            calculate_y_values(negative_avg, positive_avg, height)
        );

    }

    dbg!(&waveform);
    waveform
}

fn average_values(data: &[f32]) -> (f32, f32) {
    let positive_sum = data.iter().filter(|&x| *x > 0.0).sum::<f32>();
    let negative_sum = data.iter().filter(|&x| *x < 0.0).sum::<f32>();
    let positive_len = data.iter().filter(|&x| *x > 0.0).count();
    let negative_len = data.iter().filter(|&x| *x < 0.0).count();

    let positive_avg = positive_sum / positive_len as f32;
    let negative_avg = negative_sum / negative_len as f32;

    (positive_avg, negative_avg)
}

fn calculate_y_values(min_val: f32, max_val: f32, height: usize) -> (u16, u16) {
    let min_y = ((min_val + 1.0) / 2.0 * height as f32) as u16;
    let max_y = ((max_val + 1.0) / 2.0 * height as f32) as u16;
    let clamped_max_y = max_y.min(height as u16 - 1);
    (min_y, clamped_max_y)
}

pub fn frame_waveform(
    data: &Vec<Vec<f32>>,
    samples_per_pixel: usize,
    height: u32,
) -> Vec<Vec<(u32, u32, Rgba<u8>)>> {
    let mut pixelbuf: Vec<Vec<(u32, u32, Rgba<u8>)>> = Vec::new();
    for _ in 0..data.len() {
        let inner_vec: Vec<(u32, u32, Rgba<u8>)> = Vec::new();
        pixelbuf.push(inner_vec);
    }

    for (channel_index, channel_data) in data.iter().enumerate() {
        for (x, chunk) in channel_data.chunks_exact(samples_per_pixel).enumerate() {
            let x = x as u32;
            let color = match channel_index {
                0 => Rgba([200u8, 200u8, 200u8, 255u8]),
                _ => Rgba([180u8, 180u8, 180u8, 255u8]),
            };

            let min_val = *chunk
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(&0.0);
            let max_val = *chunk
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(&0.0);

            let min_y = ((min_val + 1.0) / 2.0 * (height as f32)) as u32;
            let max_y = ((max_val + 1.0) / 2.0 * (height as f32)) as u32;

            let clamped_max_y = max_y.min(height - 1);
            for y in min_y..=clamped_max_y {
                pixelbuf[channel_index].push((x, y, color));
            }
        }

        // Process remaining samples
        let remaining_chunk = channel_data.chunks(samples_per_pixel).last();
        if let Some(chunk) = remaining_chunk {
            let x = channel_data.len() as u32 / samples_per_pixel as u32;
            let color = match channel_index {
                0 => Rgba([200u8, 200u8, 200u8, 255u8]),
                _ => Rgba([180u8, 180u8, 180u8, 255u8]),
            };

            let min_val = *chunk
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(&0.0);
            let max_val = *chunk
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(&0.0);

            let min_y = ((min_val + 1.0) / 2.0 * (height as f32)) as u32;
            let max_y = ((max_val + 1.0) / 2.0 * (height as f32)) as u32;

            let clamped_max_y = max_y.min(height - 1);
            for y in min_y..=clamped_max_y {
                pixelbuf[channel_index].push((x, y, color));
            }
        }
    }

    pixelbuf
}

pub fn frame_spectrogram(
    pcm_data: Vec<f32>,
    fft_size: usize,
    hop_size: usize,
    percent_norm: f32,
) -> Vec<Vec<f32>> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);

    // Compute Hamming window.
    let hamming_window: Vec<f32> = (0..fft_size)
        .map(|i| {
            0.54 - 0.46 * ((2.0 * std::f32::consts::PI * i as f32) / (fft_size as f32 - 1.0)).cos()
        })
        .collect();

    process_frame(
        &mut planner,
        &fft,
        &hamming_window,
        &pcm_data,
        fft_size,
        hop_size,
        percent_norm,
    )
}

fn process_frame(
    planner: &mut FftPlanner<f32>,
    fft: &Arc<dyn rustfft::Fft<f32>>,
    hamming_window: &Vec<f32>,
    data: &Vec<f32>,
    fft_size: usize,
    hop_size: usize,
    percent_norm: f32,
) -> Vec<Vec<f32>> {
    // Initialize empty vector for storing all the processed FFT frames
    let mut all_fft_frames: Vec<Vec<f32>> = Vec::new();

    let mut start = 0;
    while start + fft_size <= data.len() {
        // Apply the window function and FFT to the current portion of data
        let mut input_output: Vec<Complex<f32>> = data[start..start + fft_size]
            .into_iter()
            .enumerate()
            .map(|(i, x)| Complex::new(x * hamming_window[i], 0.0))
            .collect();

        fft.process(&mut input_output);

        // Normalize the magnitudes to 0.0 - 1.0 range.
        let max_magnitude = input_output.iter().map(|c| c.norm()).fold(0.0, f32::max);
        if percent_norm > 0.0 {
            let normalized_magnitudes = percentile_norm(
                &input_output
                    .iter()
                    .take(fft_size / 2)
                    .map(|c| c.norm())
                    .collect::<Vec<_>>(),
                percent_norm,
            );
            all_fft_frames.push(normalized_magnitudes);
        } else {
            let normalized_magnitudes: Vec<f32> = input_output
                .iter()
                .map(|c| c.norm() / max_magnitude)
                .take(fft_size / 2)
                .collect();

            all_fft_frames.push(normalized_magnitudes);
        }
        start += hop_size;
    }

    all_fft_frames
}

fn percentile_norm(input: &[f32], percentile: f32) -> Vec<f32> {
    assert!(percentile > 0.0 && percentile <= 1.0);

    let mut sorted = input.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let index = (sorted.len() as f32 * percentile).ceil() as usize - 1;
    let norm_factor = sorted[index];

    input.iter().map(|&x| x / norm_factor).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

}

