use std::{collections::HashMap, fmt};

use crate::io_device::IoDevice;

const PAGE_SIZE_4K_OFFSET: usize = 12;
const PAGE_SIZE_4K_MASK: usize = (1 << PAGE_SIZE_4K_OFFSET) - 1;
const PAGE_SIZE_4K: usize = 1 << PAGE_SIZE_4K_OFFSET;

/// Errors that can occur during memory operations.
#[derive(Debug)]
pub enum MemoryError {
    /// Address was out of valid range.
    OutOfBounds(u64),
    /// Access was misaligned.
    Misaligned(u64),
    // No device registered at the given address.
    UnmappedAddress(u64),
    // Device exists but rejects the access.
    UnimplementedRegister(u64),
}

impl fmt::Display for MemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfBounds(addr) => write!(f, "memory access out of bounds at 0x{addr:016x}"),
            Self::Misaligned(addr) => write!(f, "misaligned memory access at 0x{addr:016x}"),
            Self::UnmappedAddress(addr) => write!(f, "unmapped address 0x{addr:016x}"),
            Self::UnimplementedRegister(addr) => {
                write!(f, "unimplemented register at 0x{addr:016x}")
            }
        }
    }
}

impl std::error::Error for MemoryError {}

#[derive(Debug, Default)]
pub struct PhysicalMemory {
    memory_map: HashMap<u64, Box<[u8; PAGE_SIZE_4K as usize]>>,
}

impl PhysicalMemory {
    pub fn new() -> Self {
        Self::default()
    }

    /// Zero `size` bytes starting at `addr`.
    ///
    /// Returns `MemoryError::OutOfBounds` if the range extends past the end
    /// of memory.
    pub fn fill_zeros(&mut self, addr: u64, size: usize) -> Result<(), MemoryError> {
        if size == 0 {
            return Ok(());
        }

        let zero_page = [0u8; PAGE_SIZE_4K as usize];
        let mut current_addr = addr;
        let mut bytes_remaining = size;
        while bytes_remaining > 0 {
            let chunk_size = bytes_remaining.min(PAGE_SIZE_4K);

            self.write_bytes(current_addr, &zero_page[..chunk_size])?;
            current_addr += chunk_size as u64;
            bytes_remaining -= chunk_size;
        }
        Ok(())
    }
}

impl IoDevice for PhysicalMemory {
    fn read_u8(&mut self, addr: u64) -> Result<u8, MemoryError> {
        let mut data = [0u8; 1];
        self.read_bytes(addr, &mut data)?;
        Ok(u8::from_le_bytes(data))
    }

    fn write_u8(&mut self, addr: u64, value: u8) -> Result<(), MemoryError> {
        self.write_bytes(addr, &[value])?;
        Ok(())
    }

    /// Read a little‑endian `u32` from the given address.
    ///
    /// Returns `MemoryError::OutOfBounds` if `addr` or `addr+3` lies
    /// outside the valid memory range.
    fn read_u32(&mut self, addr: u64) -> Result<u32, MemoryError> {
        // use read_bytes to avoid duplicating the out-of-bounds check
        let mut data = [0u8; 4];
        self.read_bytes(addr, &mut data)?;
        Ok(u32::from_le_bytes(data))
    }

    /// Write a `u32` value as four little‑endian bytes at the given address.
    ///
    /// Returns `MemoryError::OutOfBounds` if `addr` or `addr+3` lies
    /// outside the valid memory range.
    fn write_u32(&mut self, addr: u64, val: u32) -> Result<(), MemoryError> {
        self.write_bytes(addr, &val.to_le_bytes())
    }

    fn write_bytes(&mut self, addr: u64, data: &[u8]) -> Result<(), MemoryError> {
        if data.is_empty() {
            return Ok(());
        }
        let start_ppn = addr / PAGE_SIZE_4K as u64;
        let end_ppn = (addr + data.len() as u64 - 1) / PAGE_SIZE_4K as u64;
        let mut data_pos = 0usize;
        let mut bytes_remaining = data.len();
        for ppn in start_ppn..=end_ppn {
            let page_offset = if ppn == start_ppn {
                (addr & PAGE_SIZE_4K_MASK as u64) as usize
            } else {
                0
            };
            let bytes_to_copy = (PAGE_SIZE_4K - page_offset).min(bytes_remaining);
            let page_data = self
                .memory_map
                .entry(ppn)
                .or_insert_with(|| Box::new([0u8; PAGE_SIZE_4K]));
            page_data[page_offset as usize..page_offset as usize + bytes_to_copy]
                .copy_from_slice(&data[data_pos..data_pos + bytes_to_copy]);
            data_pos += bytes_to_copy;
            bytes_remaining -= bytes_to_copy;
        }
        Ok(())
    }

    fn read_bytes(&mut self, addr: u64, data: &mut [u8]) -> Result<(), MemoryError> {
        if data.is_empty() {
            return Ok(());
        }

        let start_ppn = addr / PAGE_SIZE_4K as u64;
        let end_ppn = (addr + data.len() as u64 - 1) / PAGE_SIZE_4K as u64;
        let mut data_pos = 0usize;
        let mut bytes_remaining = data.len();
        for ppn in start_ppn..=end_ppn {
            let page_offset = if ppn == start_ppn {
                (addr & PAGE_SIZE_4K_MASK as u64) as usize
            } else {
                0
            };
            let bytes_to_copy = (PAGE_SIZE_4K - page_offset).min(bytes_remaining);
            match self.memory_map.get(&ppn) {
                Some(page_data) => {
                    data[data_pos..data_pos + bytes_to_copy]
                        .copy_from_slice(&page_data[page_offset..page_offset + bytes_to_copy]);
                }
                None => {
                    return Err(MemoryError::OutOfBounds(ppn * PAGE_SIZE_4K as u64));
                }
            }
            data_pos += bytes_to_copy;
            bytes_remaining -= bytes_to_copy;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_write_u32() {
        let mut mem = PhysicalMemory::new();

        // Write and read back at address 0.
        mem.write_u32(0x00, 0xDEAD_BEEF).unwrap();
        assert_eq!(mem.read_u32(0x00).unwrap(), 0xDEAD_BEEF);

        // Write at a non‑zero aligned address.
        mem.write_u32(0x10, 0x1234_5678).unwrap();
        assert_eq!(mem.read_u32(0x10).unwrap(), 0x1234_5678);

        // Verify little‑endian byte layout.
        assert_eq!(mem.read_u32(0x00).unwrap(), 0xDEAD_BEEF);
        // Bytes at 0..4 should be EF, BE, AD, DE
        let mut bytes = [0u8; 4];
        mem.read_bytes(0x00, &mut bytes).unwrap();
        assert_eq!(bytes, [0xEF, 0xBE, 0xAD, 0xDE]);

        // Ensure writes don't clobber neighbouring regions.
        mem.write_u32(0x04, 0xAABB_CCDD).unwrap();
        assert_eq!(mem.read_u32(0x00).unwrap(), 0xDEAD_BEEF);
        assert_eq!(mem.read_u32(0x04).unwrap(), 0xAABB_CCDD);
    }

    #[test]
    fn test_write_bytes() {
        let mut mem = PhysicalMemory::new();

        let data = b"Hello!";
        mem.write_bytes(0, data).unwrap();

        // Read back as bytes.
        let mut buf = [0u8; 6];
        mem.read_bytes(0, &mut buf).unwrap();
        assert_eq!(&buf, data);

        // Write at an offset and read back.
        mem.write_bytes(10, &[0xAA, 0xBB, 0xCC]).unwrap();
        let mut verify = [0u8; 3];
        mem.read_bytes(10, &mut verify).unwrap();
        assert_eq!(verify, [0xAA, 0xBB, 0xCC]);
    }

    #[test]
    fn test_fill_zeros_multi_page() {
        let mut mem = PhysicalMemory::new();

        // Write non-zero data across three pages then zero it all.
        let size = PAGE_SIZE_4K * 2 + 512;
        let addr = 0x1000u64;
        mem.write_bytes(addr, &vec![0xFFu8; size]).unwrap();

        mem.fill_zeros(addr, size).unwrap();

        // Check first byte, a byte in the middle (page boundary), and last byte.
        let mut buf = [0xFFu8; 1];
        mem.read_bytes(addr, &mut buf).unwrap();
        assert_eq!(buf[0], 0, "first byte should be zero");

        mem.read_bytes(addr + PAGE_SIZE_4K as u64, &mut buf)
            .unwrap();
        assert_eq!(buf[0], 0, "byte at page boundary should be zero");

        mem.read_bytes(addr + size as u64 - 1, &mut buf).unwrap();
        assert_eq!(buf[0], 0, "last byte should be zero");
    }

    #[test]
    fn test_out_of_bounds_detection() {
        let mut mem = PhysicalMemory::new();

        // Read from unallocated page → OutOfBounds.
        assert!(mem.read_u32(0).is_err());
        assert!(mem.read_u32(0x400000).is_err());

        let mut mem_mut = PhysicalMemory::new();

        // Writes always succeed — pages are allocated on demand.
        assert!(mem_mut.write_u32(0, 0xDEAD_BEEF).is_ok());
        assert!(mem_mut.write_u32(0x400000, 0x1234_5678).is_ok());
        assert!(mem_mut.write_bytes(0, &[0xAA, 0xBB]).is_ok());

        // Read from an allocated page → Ok.
        assert!(mem_mut.read_u32(0).is_ok());
        assert!(mem_mut.read_u32(0x400000).is_ok());

        // Read from a page that was never written → OutOfBounds.
        assert!(mem_mut.read_u32(0x1000).is_err());

        // Empty slice always succeeds regardless of address.
        assert!(mem_mut.write_bytes(0xDEAD_BEEF_DEAD_BEEF, &[]).is_ok());
    }
}
