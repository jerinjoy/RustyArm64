use std::fmt;

use object::read::elf::ElfFile64;
use object::Object;
use object::ObjectSegment;

use crate::cpu::Cpu;

/// Errors that can occur during ELF loading.
#[derive(Debug)]
pub enum LoadError {
    /// The data is not a valid ELF file.
    NotElf,
    /// The ELF file is truncated or contains invalid data.
    Truncated,
    /// The ELF file is not for AArch64 (64-bit little-endian).
    UnsupportedArch,
    /// The ELF file has no loadable segments.
    NoLoadableSegments,
    /// A segment's file data is out of bounds.
    SegmentDataOutOfBounds,
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotElf => write!(f, "not a valid ELF file"),
            Self::Truncated => write!(f, "ELF file truncated or contains invalid data"),
            Self::UnsupportedArch => {
                write!(f, "ELF is not a 64-bit little-endian AArch64 executable")
            }
            Self::NoLoadableSegments => write!(f, "ELF has no loadable segments"),
            Self::SegmentDataOutOfBounds => write!(f, "segment file data out of bounds"),
        }
    }
}

impl std::error::Error for LoadError {}

/// The complete state of the simulated machine, including CPU and memory.
pub struct MachineState {
    /// The CPU state.
    pub cpu: Cpu,
    /// Flat byte-addressable memory.
    pub memory: Vec<u8>,
}

impl MachineState {
    /// The default memory size allocated when no segments are present (64 KB).
    const DEFAULT_MEMORY_SIZE: usize = 64 * 1024;

    /// Create a new machine state with the given memory size.
    pub fn with_memory_size(size: usize) -> Self {
        Self {
            cpu: Cpu::new(),
            memory: vec![0u8; size],
        }
    }

    /// Read a byte from the given address. Returns `None` if out of bounds.
    pub fn read_byte(&self, addr: u64) -> Option<u8> {
        self.memory.get(addr as usize).copied()
    }

    /// Write a byte to the given address. Returns `false` if out of bounds.
    pub fn write_byte(&mut self, addr: u64, value: u8) -> bool {
        if let Some(byte) = self.memory.get_mut(addr as usize) {
            *byte = value;
            true
        } else {
            false
        }
    }

    /// Read a 32-bit little-endian word from the given address.
    /// Returns `None` if any byte is out of bounds.
    pub fn read_word(&self, addr: u64) -> Option<u32> {
        let b0 = self.read_byte(addr)? as u32;
        let b1 = self.read_byte(addr + 1)? as u32;
        let b2 = self.read_byte(addr + 2)? as u32;
        let b3 = self.read_byte(addr + 3)? as u32;
        Some(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24))
    }
}

/// ELF identification constants used for pre-parse validation.
const EM_AARCH64: u16 = 0xB7;
const ELFCLASS64: u8 = 2;
const ELFDATA2LSB: u8 = 1;

/// Check whether the raw bytes represent a 64-bit little-endian AArch64 ELF.
///
/// This performs a quick header check before handing the data to the full
/// ELF parser, ensuring we only pass supported files.
fn check_elf_header(data: &[u8]) -> Result<(), LoadError> {
    // Need at least 20 bytes for the basic identification fields.
    if data.len() < 20 {
        return Err(LoadError::Truncated);
    }

    // ELF magic: 0x7F 'E' 'L' 'F'
    if data[0] != 0x7f || data[1] != b'E' || data[2] != b'L' || data[3] != b'F' {
        return Err(LoadError::NotElf);
    }

    // ELF class: 2 = 64-bit
    if data[4] != ELFCLASS64 {
        return Err(LoadError::UnsupportedArch);
    }

    // Data encoding: 1 = little-endian
    if data[5] != ELFDATA2LSB {
        return Err(LoadError::UnsupportedArch);
    }

    // e_machine field at offset 0x12 (18), 2 bytes little-endian.
    let machine = u16::from_le_bytes([data[0x12], data[0x13]]);
    if machine != EM_AARCH64 {
        return Err(LoadError::UnsupportedArch);
    }

    Ok(())
}

/// Load an AArch64 ELF executable from raw bytes.
///
/// This function parses the ELF, validates that it is a 64-bit little-endian
/// AArch64 executable, copies all `PT_LOAD` segments into a flat memory array,
/// and sets the program counter to the ELF entry point.
///
/// No relocations or dynamic linking are performed.
pub fn load_elf(data: &[u8]) -> Result<MachineState, LoadError> {
    // Quick header validation before full parse.
    check_elf_header(data)?;

    // Parse the ELF file. We specify the concrete type `Elf64<Endianness>` to
    // avoid ambiguity.
    let elf: ElfFile64<'_> =
        ElfFile64::parse(data).map_err(|_| LoadError::NotElf)?;

    // Collect loadable segments. We use `segment.data()` which only returns
    // data for loadable segments (PT_LOAD with non-zero file size).
    let mut segments: Vec<(u64, &[u8], u64)> = Vec::new();
    let mut max_addr: u64 = 0;

    for segment in elf.segments() {
        // `data()` returns an error for non-loadable or invalid segments.
        let file_data = match segment.data() {
            Ok(d) => d,
            Err(_) => continue,
        };

        let vaddr = segment.address();
        let memsz = segment.size();

        // Skip empty segments.
        if file_data.is_empty() && memsz == 0 {
            continue;
        }

        let end_addr = vaddr + memsz;
        if end_addr > max_addr {
            max_addr = end_addr;
        }

        segments.push((vaddr, file_data, memsz));
    }

    if segments.is_empty() {
        return Err(LoadError::NoLoadableSegments);
    }

    // Allocate memory large enough to cover all segments.
    let memory_size = max_addr as usize;
    let mut state =
        MachineState::with_memory_size(memory_size.max(MachineState::DEFAULT_MEMORY_SIZE));

    // Copy segment data into memory. Zero-fill the gap between file size and
    // memory size (BSS), which is already zero-initialized.
    for (vaddr, file_data, memsz) in &segments {
        let start = *vaddr as usize;
        let data_len = file_data.len();
        let mem_end = start + *memsz as usize;

        if mem_end > state.memory.len() {
            return Err(LoadError::SegmentDataOutOfBounds);
        }

        // Copy file data.
        state.memory[start..start + data_len].copy_from_slice(file_data);
        // The rest (BSS) is already zero from initialization.
        let bss_start = start + data_len;
        if bss_start < mem_end {
            // Ensure the BSS region is zeroed.
            state.memory[bss_start..mem_end].fill(0);
        }
    }

    // Set PC to the ELF entry point.
    let entry = elf.entry();
    state.cpu.write_pc(entry);

    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal AArch64 ELF executable in memory.
    ///
    /// This constructs a valid ELF64 little-endian file with:
    /// - e_machine = EM_AARCH64 (0xB7)
    /// - a single PT_LOAD segment at 0x400000
    /// - entry point = 0x400000
    /// - .text containing a RET instruction (0xD65F03C0)
    fn build_minimal_elf() -> Vec<u8> {
        let mut elf = Vec::new();

        // ELF magic + ident
        elf.extend_from_slice(&[0x7f, b'E', b'L', b'F']);
        elf.push(2); // ELFCLASS64
        elf.push(1); // ELFDATA2LSB
        elf.push(1); // EV_CURRENT
        elf.push(0); // ELFOSABI_NONE
        elf.push(0); // EI_ABIVERSION
        elf.extend_from_slice(&[0u8; 7]);

        // e_type = ET_EXEC (2)
        elf.extend_from_slice(&2u16.to_le_bytes());
        // e_machine = EM_AARCH64 (0xB7)
        elf.extend_from_slice(&0xB7u16.to_le_bytes());
        // e_version
        elf.extend_from_slice(&1u32.to_le_bytes());
        // e_entry = 0x400000
        elf.extend_from_slice(&0x400000u64.to_le_bytes());
        // e_phoff = 64
        elf.extend_from_slice(&64u64.to_le_bytes());
        // e_shoff = 0
        elf.extend_from_slice(&0u64.to_le_bytes());
        // e_flags = 0
        elf.extend_from_slice(&0u32.to_le_bytes());
        // e_ehsize = 64
        elf.extend_from_slice(&64u16.to_le_bytes());
        // e_phentsize = 56
        elf.extend_from_slice(&56u16.to_le_bytes());
        // e_phnum = 1
        elf.extend_from_slice(&1u16.to_le_bytes());
        // e_shentsize = 64
        elf.extend_from_slice(&64u16.to_le_bytes());
        // e_shnum = 0
        elf.extend_from_slice(&0u16.to_le_bytes());
        // e_shstrndx = 0
        elf.extend_from_slice(&0u16.to_le_bytes());

        // Program header (PT_LOAD)
        // p_type = PT_LOAD (1)
        elf.extend_from_slice(&1u32.to_le_bytes());
        // p_flags = PF_R | PF_W | PF_X = 4 | 2 | 1 = 7
        elf.extend_from_slice(&7u32.to_le_bytes());
        // p_offset = 0x1000
        elf.extend_from_slice(&0x1000u64.to_le_bytes());
        // p_vaddr = 0x400000
        elf.extend_from_slice(&0x400000u64.to_le_bytes());
        // p_paddr = 0x400000
        elf.extend_from_slice(&0x400000u64.to_le_bytes());
        // p_filesz = 4
        elf.extend_from_slice(&4u64.to_le_bytes());
        // p_memsz = 4
        elf.extend_from_slice(&4u64.to_le_bytes());
        // p_align = 0x1000
        elf.extend_from_slice(&0x1000u64.to_le_bytes());

        // Pad to p_offset (0x1000) with zeros
        while elf.len() < 0x1000 {
            elf.push(0);
        }

        // .text section: RET instruction (D65F03C0 in little-endian)
        elf.extend_from_slice(&[0xC0, 0x03, 0x5F, 0xD6]);

        elf
    }

    #[test]
    fn test_load_valid_elf() {
        let elf_data = build_minimal_elf();
        let state = load_elf(&elf_data).expect("should load valid ELF");

        // PC should be set to entry point.
        assert_eq!(
            state.cpu.read_pc(),
            0x400000,
            "PC should match ELF entry point"
        );

        // Check that memory at 0x400000 contains the RET instruction.
        let word = state
            .read_word(0x400000)
            .expect("address 0x400000 should be in memory");
        assert_eq!(word, 0xD65F03C0, "memory should contain the RET instruction");
    }

    #[test]
    fn test_load_elf_memory_coverage() {
        let elf_data = build_minimal_elf();
        let state = load_elf(&elf_data).expect("should load valid ELF");

        // Memory should be at least large enough to cover the segment.
        assert!(
            state.memory.len() >= 0x400000 + 4,
            "memory should cover the loaded segment"
        );
    }

    #[test]
    fn test_load_truncated_file() {
        // Just a few bytes — not a valid ELF.
        let data = [0x7fu8, b'E', b'L', b'F'];
        let result = load_elf(&data);
        assert!(result.is_err(), "truncated ELF should produce an error");
    }

    #[test]
    fn test_load_non_elf_file() {
        let data = [0u8; 128];
        let result = load_elf(&data);
        assert!(result.is_err(), "non-ELF data should produce an error");
    }

    #[test]
    fn test_load_error_display() {
        let err = LoadError::NotElf;
        assert_eq!(format!("{}", err), "not a valid ELF file");

        let err = LoadError::UnsupportedArch;
        assert_eq!(
            format!("{}", err),
            "ELF is not a 64-bit little-endian AArch64 executable"
        );
    }

    #[test]
    fn test_machine_state_read_write() {
        let mut state = MachineState::with_memory_size(1024);

        // Write a byte
        assert!(state.write_byte(0x100, 0xAB));
        assert_eq!(state.read_byte(0x100), Some(0xAB));

        // Read word from written bytes
        state.write_byte(0x200, 0x78);
        state.write_byte(0x201, 0x56);
        state.write_byte(0x202, 0x34);
        state.write_byte(0x203, 0x12);
        assert_eq!(state.read_word(0x200), Some(0x12345678));

        // Out of bounds
        assert_eq!(state.read_byte(2048), None);
        assert!(!state.write_byte(2048, 0xFF));
    }
}
