use std::fmt;

use goblin::elf::Elf;
use goblin::elf::header;
use goblin::elf::program_header;

use crate::memory::PhysicalMemory;

use crate::io_device::IoDevice;

/// Errors that can occur during ELF loading.
#[derive(Debug, PartialEq, Eq)]
pub enum LoaderError {
    /// The data is not a valid ELF file (bad magic or parse failure).
    NotElf,
    /// The ELF file is truncated or contains invalid data.
    Truncated,
    /// The ELF file is not a 64-bit little-endian AArch64 executable.
    UnsupportedArch,
    /// The ELF file has no loadable segments.
    NoLoadableSegments,
    /// A segment's file data is out of bounds.
    SegmentDataOutOfBounds,
}

impl fmt::Display for LoaderError {
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

impl std::error::Error for LoaderError {}

/// Load an AArch64 ELF executable from raw bytes into the given memory.
///
/// This function parses the ELF, validates that it is a 64-bit little-endian
/// AArch64 executable, copies all `PT_LOAD` segments into memory via
/// [`PhysicalMemory::write_bytes`], and returns the ELF entry point address.
///
/// No relocations or dynamic linking are performed.
pub fn load_elf(memory: &mut PhysicalMemory, elf_bytes: &[u8]) -> Result<u64, LoaderError> {
    // Quick pre-check: ELF magic + minimum size for identification fields.
    if elf_bytes.len() < 20 {
        return Err(LoaderError::Truncated);
    }
    if elf_bytes[0..4] != [0x7f, b'E', b'L', b'F'] {
        return Err(LoaderError::NotElf);
    }
    // Reject 32-bit ELF early (EI_CLASS at offset 4).
    if elf_bytes[4] != 2 {
        return Err(LoaderError::UnsupportedArch);
    }
    // Reject big-endian early (EI_DATA at offset 5).
    if elf_bytes[5] != 1 {
        return Err(LoaderError::UnsupportedArch);
    }

    // Full parse with goblin.
    let elf = Elf::parse(elf_bytes).map_err(|_| LoaderError::NotElf)?;

    // Double-check machine field.
    if elf.header.e_machine != header::EM_AARCH64 {
        return Err(LoaderError::UnsupportedArch);
    }

    let phdrs = &elf.program_headers;
    if phdrs.is_empty() {
        return Err(LoaderError::NoLoadableSegments);
    }

    let mut any_loaded = false;

    for phdr in phdrs {
        if phdr.p_type != program_header::PT_LOAD {
            continue;
        }

        let file_sz = phdr.p_filesz;
        let mem_sz = phdr.p_memsz;
        let vaddr = phdr.p_vaddr;

        if file_sz == 0 && mem_sz == 0 {
            continue;
        }

        // Copy file data into memory.
        if file_sz > 0 {
            let start = phdr.p_offset as usize;
            let end = start
                .checked_add(file_sz as usize)
                .ok_or(LoaderError::SegmentDataOutOfBounds)?;
            if end > elf_bytes.len() {
                return Err(LoaderError::SegmentDataOutOfBounds);
            }
            let segment_data = &elf_bytes[start..end];
            memory
                .write_bytes(vaddr, segment_data)
                .map_err(|_| LoaderError::SegmentDataOutOfBounds)?;
        }

        // Zero-fill BSS region (where mem_sz > file_sz).
        if mem_sz > file_sz {
            let bss_start = vaddr
                .checked_add(file_sz)
                .ok_or(LoaderError::SegmentDataOutOfBounds)?;
            let bss_size = (mem_sz - file_sz) as usize;
            memory
                .fill_zeros(bss_start, bss_size)
                .map_err(|_| LoaderError::SegmentDataOutOfBounds)?;
        }

        any_loaded = true;
    }

    if !any_loaded {
        return Err(LoaderError::NoLoadableSegments);
    }

    Ok(elf.header.e_entry)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ELF builders ─────────────────────────────────────────────────

    /// Build a minimal valid AArch64 ELF64 executable.
    ///
    /// * entry = `entry`
    /// * one PT_LOAD segment at `entry` containing `segment_bytes`
    fn build_elf(entry: u64, segment_bytes: &[u8]) -> Vec<u8> {
        let hdr_size: u64 = 64;
        let phdr_size: u64 = 56;
        let phoff: u64 = hdr_size;
        // Place segment data immediately after the program header.
        let file_offset: u64 = hdr_size + phdr_size;

        let filesz = segment_bytes.len() as u64;
        let memsz = filesz; // no BSS for simplicity

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
        elf.extend_from_slice(&header::EM_AARCH64.to_le_bytes());
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
        elf.extend_from_slice(segment_bytes);

        elf
    }

    /// Build an ELF identical to `build_elf` but with a different e_machine.
    fn build_elf_with_machine(entry: u64, segment_bytes: &[u8], machine: u16) -> Vec<u8> {
        let mut elf = build_elf(entry, segment_bytes);
        // e_machine is at offset 0x12 (18)
        elf[18..20].copy_from_slice(&machine.to_le_bytes());
        elf
    }

    /// Build an ELF that is marked as 32-bit (ELFCLASS32 = 1).
    fn build_32bit_elf(entry: u64, segment_bytes: &[u8]) -> Vec<u8> {
        let mut elf = build_elf(entry, segment_bytes);
        // EI_CLASS at offset 4
        elf[4] = 1; // ELFCLASS32
        elf
    }

    // ── Tests ────────────────────────────────────────────────────────

    #[test]
    fn test_load_valid_arm64_elf() {
        // RET instruction (D65F03C0 in little-endian).
        let segment = [0xC0u8, 0x03, 0x5F, 0xD6];
        let entry = 0x400000u64;
        let elf_bytes = build_elf(entry, &segment);

        // Allocate enough memory to cover the segment.
        let mut memory = PhysicalMemory::new();

        let result = load_elf(&mut memory, &elf_bytes);
        assert!(result.is_ok(), "valid AArch64 ELF should load successfully");
        assert_eq!(result.unwrap(), entry);

        // Verify segment data was written to memory.
        let word = memory
            .read_u32(entry)
            .expect("entry address should be in bounds");
        assert_eq!(
            word, 0xD65F03C0,
            "memory should contain the RET instruction"
        );
    }

    #[test]
    fn test_reject_non_arm64() {
        // Build an x86-64 ELF (EM_X86_64 = 0x3E).
        let segment = [0x90u8; 4]; // NOP sled
        let elf_bytes = build_elf_with_machine(0x400000, &segment, 0x3E);

        let mut memory = PhysicalMemory::new();
        let result = load_elf(&mut memory, &elf_bytes);
        assert!(result.is_err(), "non-ARM64 ELF should be rejected");
        assert_eq!(result.unwrap_err(), LoaderError::UnsupportedArch);
    }

    #[test]
    fn test_reject_32bit_elf() {
        let segment = [0x00u8; 4];
        let elf_bytes = build_32bit_elf(0x10000, &segment);

        let mut memory = PhysicalMemory::new();
        let result = load_elf(&mut memory, &elf_bytes);
        assert!(result.is_err(), "32-bit ELF should be rejected");
        assert_eq!(result.unwrap_err(), LoaderError::UnsupportedArch);
    }

    #[test]
    fn test_entry_point_correct() {
        let segment = [0xC0u8, 0x03, 0x5F, 0xD6];
        let entry = 0x8000_0000u64;
        let elf_bytes = build_elf(entry, &segment);

        let mut memory = PhysicalMemory::new(); // large enough
        let got_entry = load_elf(&mut memory, &elf_bytes).expect("valid ELF should load");
        assert_eq!(got_entry, entry, "entry point should match e_entry field");
    }

    #[test]
    fn test_load_truncated_file() {
        // Just a few bytes — not a valid ELF.
        let data = [0x7fu8, b'E', b'L', b'F'];
        let mut memory = PhysicalMemory::new();
        let result = load_elf(&mut memory, &data);
        assert!(result.is_err(), "truncated ELF should produce an error");
    }

    #[test]
    fn test_load_non_elf_file() {
        let data = [0u8; 128];
        let mut memory = PhysicalMemory::new();
        let result = load_elf(&mut memory, &data);
        assert!(result.is_err(), "non-ELF data should produce an error");
        assert_eq!(result.unwrap_err(), LoaderError::NotElf);
    }

    #[test]
    fn test_load_error_display() {
        assert_eq!(format!("{}", LoaderError::NotElf), "not a valid ELF file");
        assert_eq!(
            format!("{}", LoaderError::UnsupportedArch),
            "ELF is not a 64-bit little-endian AArch64 executable"
        );
        assert_eq!(
            format!("{}", LoaderError::NoLoadableSegments),
            "ELF has no loadable segments"
        );
        assert_eq!(
            format!("{}", LoaderError::SegmentDataOutOfBounds),
            "segment file data out of bounds"
        );
        assert_eq!(
            format!("{}", LoaderError::Truncated),
            "ELF file truncated or contains invalid data"
        );
    }
}
