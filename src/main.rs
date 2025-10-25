#![feature(portable_simd)]

use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use rayon::prelude::*;
use clap::Parser;
use std::simd::{Simd};
use std::simd::prelude::*;
use ocl::{Platform, Device, Context, Queue, Program, Buffer, Kernel, flags};


#[inline]
fn apply_rule(val: u8) -> u8 {
    match val {
        7 | 4 | 0 => 0,
        _ => 1,
    }
}

fn simulate_rayon(input: &[u8], out: &mut [u8]) {
    let end = input.len().saturating_sub(2);

    out[1..=end].par_iter_mut().enumerate().for_each(|(idx, out_elem)| {
        let i = idx + 1;
        let cc = (input[i - 1] << 2) | (input[i] << 1) | input[i + 1];
        *out_elem = apply_rule(cc);
    });
}

fn simulate(input: &[u8], out: &mut [u8], indices: &[usize]) {
    let end = input.len().saturating_sub(2);
    for (o, idx) in out[1..=end].iter_mut().zip(indices.iter().copied()) {
        let cc = (input[idx - 1] << 2) | (input[idx] << 1) | input[idx + 1];
        *o = apply_rule(cc);
    }
}

pub fn simulate_simd(input: &[u8], out: &mut [u8]) {
    let n = input.len();
    if n < 3 {
        out.copy_from_slice(input);
        return;
    }

    type V = Simd<u8, 16>;
    let lanes = V::splat(0).len(); // number of lanes
    let mut i = 1;

    while i + lanes <= n - 1 {
        let left  = V::from_slice(&input[i - 1 .. i - 1 + lanes]);
        let mid   = V::from_slice(&input[i     .. i     + lanes]);
        let right = V::from_slice(&input[i + 1 .. i + 1 + lanes]);

        let cc = (left << Simd::splat(2)) | (mid << Simd::splat(1)) | right;

        let mask = cc.simd_eq(Simd::splat(7))
            | cc.simd_eq(Simd::splat(4))
            | cc.simd_eq(Simd::splat(0));

        let val = mask.select(Simd::splat(0u8), Simd::splat(1u8));
        val.copy_to_slice(&mut out[i .. i + lanes]);
        i += lanes;
    }

    // scalar tail
    for j in i..n - 1 {
        let cc = (input[j - 1] << 2) | (input[j] << 1) | input[j + 1];
        out[j] = apply_rule(cc);
    }
}

const KERNEL_SRC: &str = r#"
__kernel void simulate_transform(
    __global const uchar* input,
    __global uchar* output,
    const uint n)
{
    uint i = get_global_id(0) + 1;
    if (i >= n - 1) return;

    uchar cc = (input[i - 1] << 2) | (input[i] << 1) | input[i + 1];
    uchar val = (cc == 7 || cc == 4 || cc == 0) ? 0 : 1;
    output[i] = val;
}
"#;


pub fn simulate_ocl(input: &[u8], out: &mut [u8], iterations: usize) -> ocl::Result<()> {
    assert_eq!(input.len(), out.len());
    let n = input.len();

    let platform = Platform::default();
    let device = Device::first(platform)?;

    let context = Context::builder()
        .platform(platform)
        .devices(device.clone())
        .build()?;

    let queue = Queue::new(&context, device, None)?;

    let program = Program::builder()
        .src(KERNEL_SRC)
        .devices(device)
        .build(&context)?;

    // Create working buffers
    let mut current = input.to_vec();
    let mut next = vec![0u8; n];

    for _ in 0..iterations {
        let input_buf = Buffer::<u8>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR)
            .len(n)
            .copy_host_slice(&current)
            .build()?;

        let output_buf = Buffer::<u8>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_WRITE_ONLY)
            .len(n)
            .build()?;

        let kernel = Kernel::builder()
            .program(&program)
            .name("simulate_transform")
            .queue(queue.clone())
            .global_work_size(n.saturating_sub(2))
            .arg(&input_buf)
            .arg(&output_buf)
            .arg(&(n as u32))
            .build()?;

        unsafe { kernel.enq()?; }

        output_buf.read(&mut next).enq()?;
        queue.finish()?;

        // Swap buffers for next iteration
        std::mem::swap(&mut current, &mut next);
    }

    out.copy_from_slice(&current);
    Ok(())
}

fn read_input_file<P: AsRef<Path>>(p: P) -> Vec<u8> {
    let p = p.as_ref();
    if !p.exists() {
        eprintln!("Invalid file.");
        std::process::exit(1);
    }
    let f = File::open(p).unwrap_or_else(|_| {
        eprintln!("Could not open file for reading.");
        std::process::exit(1);
    });
    let mut buf = String::new();
    let mut reader = BufReader::new(&f);
    reader.read_line(&mut buf).unwrap();
    let size: usize = buf.trim().parse().unwrap_or_else(|_| {
        eprintln!("Invalid size line.");
        std::process::exit(1);
    });

    // read next `size` bytes
    let mut raw = vec![0u8; size];
    reader.read_exact(&mut raw).unwrap_or_else(|_| {
        eprintln!("Could not read initial automata.");
        std::process::exit(1);
    });

    // sanitize: last '1'
    let pos = raw.iter().rposition(|&b| b == b'1');
    if pos.is_none() {
        println!("0");
        std::process::exit(1);
    }
    let size = pos.unwrap() + 2;
    if size > raw.len() {
        println!("Invalid input file.");
        std::process::exit(1);
    }
    raw.truncate(size);

    let mut res = Vec::with_capacity(size);
    for &ch in &raw {
        res.push(ch.saturating_sub(b'0'));
    }
    res
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
enum SimulationType {
    #[value(name = "policy")]
    Policy,
    #[value(name = "rayon")]
    Rayon,
    #[value(name = "simd")]
    Simd,
    #[value(name = "ocl")]
    Ocl,
}

#[derive(Parser, Debug)]
#[command(name = "rule110")]
#[command(about = "Rule 110 cellular automaton simulator")]
struct Args {
    /// Number of simulation steps
    #[arg(long)]
    iter: usize,

    /// Initial configuration file
    #[arg(long)]
    init: String,

    /// Simulation version (policy or rayon)
    #[arg(long, value_enum)]
    version: SimulationType,
}


fn main() {
    let args = Args::parse();

    let initial_vec = read_input_file(&args.init);
    let mut inbuf = initial_vec;
    let mut outbuf = vec![0u8; inbuf.len()];

    match args.version {
        SimulationType::Policy => {
            let end = inbuf.len().saturating_sub(2);
            let indices: Vec<usize> = (1..=end).collect();
            for _ in 0..args.iter {
                simulate(&inbuf, &mut outbuf, &indices);
                inbuf = outbuf.clone();
            }
        }
        SimulationType::Rayon => {
            for _ in 0..args.iter {
                simulate_rayon(&inbuf, &mut outbuf);
                inbuf = outbuf.clone();
            }
        }
        SimulationType::Simd => {
            for _ in 0..args.iter {
                simulate_simd(&inbuf, &mut outbuf);
                inbuf = outbuf.clone();
            }
        }
        SimulationType::Ocl => {
            simulate_ocl(&inbuf, &mut outbuf, args.iter).unwrap();
            inbuf = outbuf.clone();
        }
    }

    let ones = inbuf.par_iter().filter(|&&v| v == 1).count();
    println!("{ones}");
}
