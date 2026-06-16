use std::env;
use std::fs;
use std::process;

use arm64_simulator::cpu::Cpu;
use arm64_simulator::loader::load_elf;
use arm64_simulator::memory::Memory;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <ELF file>", args[0]);
        process::exit(1);
    }

    let elf_path = &args[1];

    // Read the ELF file.
    let elf_bytes = match fs::read(elf_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Error reading {}: {}", elf_path, e);
            process::exit(1);
        }
    };

    // Allocate memory (16 MiB for MVP).
    let mem_size: usize = 16 * 1024 * 1024;
    let mut memory = Memory::new(mem_size);

    // Load the ELF into memory.
    let entry_point = match load_elf(&mut memory, &elf_bytes) {
        Ok(addr) => addr,
        Err(e) => {
            eprintln!("Error loading ELF: {}", e);
            process::exit(1);
        }
    };

    // Create CPU and set program counter to entry point.
    let mut cpu = Cpu::new(memory);
    cpu.write_pc(entry_point);

    // Run the program.
    match cpu.run() {
        Ok(()) => {
            println!("\nProgram halted successfully.\n");
            cpu.print_registers();
        }
        Err(e) => {
            eprintln!("\nCPU error: {}\n", e);
            cpu.print_registers();
            process::exit(1);
        }
    }
}
