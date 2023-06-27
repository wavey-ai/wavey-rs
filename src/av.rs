use image::{DynamicImage, GrayImage, ImageBuffer, Luma, Rgba, RgbaImage};
use imageproc::contrast::equalize_histogram;
use rustfft::algorithm::Radix4;
use rustfft::num_traits::Zero;
use rustfft::{num_complex::Complex, Fft};
use rusttype::{Font, Scale};
use std::cmp;
use std::io::Cursor;
use std::path::Path;

use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread;

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

pub fn u16vec_to_bytes(data: &[u16]) -> Vec<u8> {
    let total_elements = data.len();
    let buffer_size = total_elements * std::mem::size_of::<u16>();
    let mut buf: Vec<u8> = Vec::with_capacity(buffer_size);

    for x in data {
        let x_bytes = u16::to_le_bytes(*x);
        buf.extend_from_slice(&x_bytes);
    }

    buf
}

pub fn process_waveform_frame(
    pcm_data: &Vec<f32>,
    fft_size: usize,
    hop_size: usize,
    samples_per_point: usize,
    hamming_window: Vec<f32>,
) -> Vec<u8> {
    // Initialize empty vectors for storing the processed FFT frames and waveform
    let mut all_fft_frames: Vec<Vec<f32>> = Vec::new();
    let mut waveform: Vec<u8> = Vec::new();

    let max_value = 255.0; // Maximum value for u8

    let mut start = 0;
    while start + fft_size <= pcm_data.len() {
        // Apply the window function to the current portion of data
        let windowed_data: Vec<f32> = pcm_data[start..start + fft_size]
            .iter()
            .zip(&hamming_window)
            .map(|(x, w)| x * w)
            .collect();

        for chunk in windowed_data.chunks(samples_per_point) {
            let mut max_value = f32::NEG_INFINITY;
            let mut min_value = f32::INFINITY;

            for &value in chunk {
                if value > max_value {
                    max_value = value;
                }
                if value < min_value {
                    min_value = value;
                }
            }

            // Scale the maximum and minimum values to the range -255 to 255
            let scaled_max = (max_value * 255.0).clamp(-255.0, 255.0) as i16;
            let scaled_min = (min_value * 255.0).clamp(-255.0, 255.0) as i16;

            // Convert the scaled values to u8 with sign encoding
            let encoded_max = if scaled_max >= 0 {
                (scaled_max as u8) | 0b1000_0000
            } else {
                ((-scaled_max) as u8) & 0b0111_1111
            };

            let encoded_min = if scaled_min >= 0 {
                (scaled_min as u8) | 0b1000_0000
            } else {
                ((-scaled_min) as u8) & 0b0111_1111
            };

            // Push the encoded maximum and minimum values to the waveform
            waveform.push(encoded_max);
            waveform.push(encoded_min);
        }
        start += hop_size;
    }

    waveform
}

pub fn process_sonograph_frame(
    pcm_data: &Vec<f32>,
    fft_size: usize,
    hop_size: usize,
    hamming_window: Vec<f32>,
    fft: &Arc<dyn Fft<f32>>,
) -> Vec<Vec<f32>> {
    let mut fft_frames: Vec<Vec<f32>> = Vec::new();

    let mut start = 0;
    while start + fft_size <= pcm_data.len() {
        // Apply the window function to the current portion of data
        let windowed_data: Vec<f32> = pcm_data[start..start + fft_size]
            .iter()
            .zip(&hamming_window)
            .map(|(x, w)| x * w)
            .collect();

        // Apply FFT to the windowed data for spectrogram calculation
        let mut input_output: Vec<Complex<f32>> = windowed_data
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();

        fft.process(&mut input_output);

        // Normalize the magnitudes to 0.0 - 1.0 range for spectrogram
        let max_magnitude = input_output.iter().map(|c| c.norm()).fold(0.0, f32::max);
        let normalized_magnitudes: Vec<f32> = input_output
            .iter()
            .map(|c| c.norm() / max_magnitude)
            .take(fft_size / 2)
            .collect();

        fft_frames.push(normalized_magnitudes);

        start += hop_size;
    }

    fft_frames
}

#[cfg(test)]
mod tests {
    use super::*;
}
