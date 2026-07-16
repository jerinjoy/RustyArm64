use arm64_simulator::cpu::Cpu;
use arm64_simulator::loader::load_elf;
use arm64_simulator::memory::Memory;

/// Build a minimal ARM64 ELF64 executable with the given instructions.
///
/// The ELF has one PT_LOAD segment at virtual address `0x400000`
/// containing `seg_bytes`. The entry point is set to `0x400000`.
///
/// See the loader unit tests for the same builder pattern.
fn build_minimal_elf(seg_bytes: &[u8]) -> Vec<u8> {
    let hdr_size: u64 = 64;
    let phdr_size: u64 = 56;
    let phoff: u64 = hdr_size;
    let file_offset: u64 = hdr_size + phdr_size; // 120 = 0x78

    let entry: u64 = 0x400000;
    let filesz = seg_bytes.len() as u64;
    let memsz = filesz; // no BSS

    let mut elf = Vec::new();

    // ── ELF header ──────────────────────────────────────────
    // e_ident
    elf.extend_from_slice(&[0x7f, b'E', b'L', b'F']); // magic
    elf.push(2); // ELFCLASS64
    elf.push(1); // ELFDATA2LSB
    elf.push(1); // EV_CURRENT
    elf.push(0); // ELFOSABI_NONE
    elf.push(0); // EI_ABIVERSION
    elf.extend_from_slice(&[0u8; 7]); // padding

    // e_type = ET_EXEC (2)
    elf.extend_from_slice(&2u16.to_le_bytes());
    // e_machine = EM_AARCH64 (0xB7)
    elf.extend_from_slice(&0xB7u16.to_le_bytes());
    // e_version
    elf.extend_from_slice(&1u32.to_le_bytes());
    // e_entry
    elf.extend_from_slice(&entry.to_le_bytes());
    // e_phoff
    elf.extend_from_slice(&phoff.to_le_bytes());
    // e_shoff
    elf.extend_from_slice(&0u64.to_le_bytes());
    // e_flags
    elf.extend_from_slice(&0u32.to_le_bytes());
    // e_ehsize
    elf.extend_from_slice(&(hdr_size as u16).to_le_bytes());
    // e_phentsize
    elf.extend_from_slice(&(phdr_size as u16).to_le_bytes());
    // e_phnum = 1
    elf.extend_from_slice(&1u16.to_le_bytes());
    // e_shentsize
    elf.extend_from_slice(&64u16.to_le_bytes());
    // e_shnum = 0
    elf.extend_from_slice(&0u16.to_le_bytes());
    // e_shstrndx = 0
    elf.extend_from_slice(&0u16.to_le_bytes());

    // ── Program header (PT_LOAD) ────────────────────────────
    // p_type = PT_LOAD (1)
    elf.extend_from_slice(&1u32.to_le_bytes());
    // p_flags = PF_R | PF_W | PF_X = 7
    elf.extend_from_slice(&7u32.to_le_bytes());
    // p_offset
    elf.extend_from_slice(&file_offset.to_le_bytes());
    // p_vaddr
    elf.extend_from_slice(&entry.to_le_bytes());
    // p_paddr
    elf.extend_from_slice(&entry.to_le_bytes());
    // p_filesz
    elf.extend_from_slice(&filesz.to_le_bytes());
    // p_memsz
    elf.extend_from_slice(&memsz.to_le_bytes());
    // p_align
    elf.extend_from_slice(&0x1000u64.to_le_bytes());

    // ── Segment data ────────────────────────────────────────
    elf.extend_from_slice(seg_bytes);

    elf
}

#[test]
fn test_elf_execution() {
    // Build the instructions:
    //   MOVZ x0, #0x42   → 0xD2800840
    //   ADD  x1, x0, #1  → 0x91000401
    //   HLT  #0          → 0xD4030000
    //
    // Little-endian bytes for each instruction:
    let movz_bytes = 0xD280_0840u32.to_le_bytes(); // [0x40, 0x08, 0x80, 0xD2]
    let add_bytes = 0x9100_0401u32.to_le_bytes(); // [0x01, 0x04, 0x00, 0x91]
    let hlt_bytes = 0xD403_0000u32.to_le_bytes(); // [0x00, 0x00, 0x03, 0xD4]

    let mut seg: Vec<u8> = Vec::new();
    seg.extend_from_slice(&movz_bytes);
    seg.extend_from_slice(&add_bytes);
    seg.extend_from_slice(&hlt_bytes);

    let elf_bytes = build_minimal_elf(&seg);

    // Allocate memory large enough: entry (0x400000) + segment (12 bytes) + margin.
    let mut memory = Memory::new(0x410000);

    // Load the ELF.
    let entry = load_elf(&mut memory, &elf_bytes).expect("ELF should load successfully");
    assert_eq!(entry, 0x400000, "entry point should be 0x400000");

    // Create CPU and set PC to entry.
    let mut cpu = Cpu::new(memory);
    cpu.write_pc(entry);

    // Run until halt.
    let result = cpu.run();
    assert!(
        result.is_ok(),
        "program should halt cleanly, got {:?}",
        result
    );
    assert!(cpu.halted, "CPU should be halted");

    // Verify register state.
    assert_eq!(cpu.read_reg(0), 0x42, "X0 should be 0x42");
    assert_eq!(cpu.read_reg(1), 0x43, "X1 should be 0x43 (X0 + 1)");

    // PC should be entry + 8 (two 4-byte instructions).
    assert_eq!(cpu.read_pc(), entry + 8, "PC should be entry + 12");
}
