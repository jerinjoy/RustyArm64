use std::fmt;

/// Errors that can occur during memory operations.
#[derive(Debug)]
pub enum MemoryError {
    /// Address was out of valid range.
    OutOfBounds(u64),
    /// Access was misaligned.
    Misaligned(u64),
}

impl fmt::Display for MemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfBounds(addr) => write!(f, "memory access out of bounds at 0x{addr:016x}"),
            Self::Misaligned(addr) => write!(f, "misaligned memory access at 0x{addr:016x}"),
        }
    }
}

impl std::error::Error for MemoryError {}

/// Flat byte‑addressable memory model.
///
/// Stores memory as a `Vec<u8>` and provides helpers for little‑endian
/// 32‑bit reads and writes with bounds checking.
#[derive(Debug)]
pub struct Memory {
    data: Vec<u8>,
}

impl Memory {
    /// Create a new zero‑initialised memory of `size` bytes.
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0u8; size],
        }
    }

    /// Return the size of this memory in bytes.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Return `true` if the memory contains zero bytes.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Read a little‑endian `u32` from the given address.
    ///
    /// Returns `MemoryError::OutOfBounds` if `addr` or `addr+3` lies
    /// outside the valid memory range.
    pub fn read_u32(&self, addr: u64) -> Result<u32, MemoryError> {
        let end = addr.checked_add(3).ok_or(MemoryError::OutOfBounds(addr))?;
        if end as usize >= self.data.len() {
            return Err(MemoryError::OutOfBounds(addr));
        }

        let base = addr as usize;
        Ok(u32::from_le_bytes(
            self.data[base..base + 4].try_into().unwrap(),
        ))
    }

    /// Write a `u32` value as four little‑endian bytes at the given address.
    ///
    /// Returns `MemoryError::OutOfBounds` if `addr` or `addr+3` lies
    /// outside the valid memory range.
    pub fn write_u32(&mut self, addr: u64, val: u32) -> Result<(), MemoryError> {
        let end = addr.checked_add(3).ok_or(MemoryError::OutOfBounds(addr))?;
        if end as usize >= self.data.len() {
            return Err(MemoryError::OutOfBounds(addr));
        }

        let base = addr as usize;
        Ok(self.data[base..base + 4].copy_from_slice(&val.to_le_bytes()))
    }

    /// Write an arbitrary slice of bytes starting at the given address.
    ///
    /// Returns `MemoryError::OutOfBounds` if the write would extend past the
    /// end of memory.
    pub fn write_bytes(&mut self, addr: u64, data: &[u8]) -> Result<(), MemoryError> {
        let len_u64 = data.len() as u64;
        let end = addr
            .checked_add(len_u64)
            .ok_or(MemoryError::OutOfBounds(addr))?;
        // `end` is one past the last byte; we need `end - 1` to be a valid index.
        if data.is_empty() {
            // Nothing to write — succeed regardless of address.
            return Ok(());
        }
        let last = end - 1;
        if last as usize >= self.data.len() {
            return Err(MemoryError::OutOfBounds(addr));
        }

        let base = addr as usize;
        self.data[base..base + data.len()].copy_from_slice(data);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_write_u32() {
        let mut mem = Memory::new(256);

        // Write and read back at address 0.
        mem.write_u32(0x00, 0xDEAD_BEEF).unwrap();
        assert_eq!(mem.read_u32(0x00).unwrap(), 0xDEAD_BEEF);

        // Write at a non‑zero aligned address.
        mem.write_u32(0x10, 0x1234_5678).unwrap();
        assert_eq!(mem.read_u32(0x10).unwrap(), 0x1234_5678);

        // Verify little‑endian byte layout.
        assert_eq!(mem.read_u32(0x00).unwrap(), 0xDEAD_BEEF);
        // Bytes at 0..4 should be EF, BE, AD, DE
        let bytes: Vec<u8> = (0..4).map(|i| mem.data[i]).collect();
        assert_eq!(bytes, vec![0xEF, 0xBE, 0xAD, 0xDE]);

        // Ensure writes don't clobber neighbouring regions.
        mem.write_u32(0x04, 0xAABB_CCDD).unwrap();
        assert_eq!(mem.read_u32(0x00).unwrap(), 0xDEAD_BEEF);
        assert_eq!(mem.read_u32(0x04).unwrap(), 0xAABB_CCDD);
    }

    #[test]
    fn test_write_bytes() {
        let mut mem = Memory::new(64);

        let data = b"Hello!";
        mem.write_bytes(0, data).unwrap();

        // Read back as bytes.
        let mut buf = [0u8; 6];
        buf.copy_from_slice(&mem.data[0..6]);
        assert_eq!(&buf, data);

        // Write at an offset.
        mem.write_bytes(10, &[0xAA, 0xBB, 0xCC]).unwrap();
        assert_eq!(mem.data[10], 0xAA);
        assert_eq!(mem.data[11], 0xBB);
        assert_eq!(mem.data[12], 0xCC);
    }

    #[test]
    fn test_out_of_bounds_detection() {
        let mem = Memory::new(16);

        // Read exactly at the boundary — 16 bytes means max index 15.
        // Read at addr 12 is ok (12,13,14,15).
        assert!(mem.read_u32(12).is_ok());
        // Read at addr 13 overflows (13+3=16, out of bounds).
        assert!(mem.read_u32(13).is_err());
        // Read far past end.
        assert!(mem.read_u32(100).is_err());

        let mut mem_mut = Memory::new(16);

        // Write exactly at the boundary.
        assert!(mem_mut.write_u32(12, 0).is_ok());
        // Write past end.
        assert!(mem_mut.write_u32(13, 0).is_err());
        assert!(mem_mut.write_u32(100, 0).is_err());

        // write_bytes boundary checks.
        assert!(mem_mut.write_bytes(15, &[0x01]).is_ok()); // last byte
        assert!(mem_mut.write_bytes(16, &[0x01]).is_err()); // one past end
        assert!(mem_mut.write_bytes(0, &[0u8; 17]).is_err()); // too large

        // Empty slice always succeeds.
        assert!(mem_mut.write_bytes(200, &[]).is_ok());
    }
}
