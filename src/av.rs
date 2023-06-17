use rustfft::algorithm::Radix4;
use rustfft::num_traits::Zero;
use rustfft::{num_complex::Complex, FftPlanner};
use std::sync::Arc;

use image::codecs::png::PngEncoder;
use image::{DynamicImage, GrayImage, ImageBuffer, Luma, Rgba, RgbaImage};
use imageproc::contrast::equalize_histogram;
use rusttype::{Font, Scale};
use std::cmp;
use std::io::Cursor;

pub fn save_spectrogram(frames: &Vec<Vec<Vec<f32>>>, eq: bool, fileprefix: &str, sample_rate: u32) {
    let width = frames[0].len();
    let height = frames[0][0].len();

    for i in 0..frames.len() {
        let tile_width = 10000; // Width of each tile in pixels
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

            let name: String = format!("{}{}_{}.png", fileprefix, i, start_pixel);
            imgbuf.save(name).unwrap();

            start_pixel += tile_width;
        }
    }
}

pub fn save_waveform(data: &Vec<Vec<f32>>, width: u32, height: u32, filename: &str) {
    let mut imgbuf: RgbaImage = ImageBuffer::new(width, height);
    for (channel_index, channel_data) in data.iter().enumerate() {
        // the number of audio samples that will be represented by each column of pixels
        let samples_per_pixel = channel_data.len() / width as usize;

        for (x, chunk) in channel_data.chunks(samples_per_pixel).enumerate() {
            let x = x as u32;
            if x >= width {
                break;
            }

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

            for y in min_y..=max_y {
                imgbuf.put_pixel(x, y, color);
            }
        }
    }

    imgbuf.save(filename).unwrap();
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
