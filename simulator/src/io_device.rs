use crate::memory::MemoryError;

pub trait IoDevice {
    fn read_u8(&mut self, addr: u64) -> Result<u8, MemoryError>;
    fn write_u8(&mut self, addr: u64, value: u8) -> Result<(), MemoryError>;
    // fn read_u16(&mut self, addr: u64) -> Result<u16, MemoryError>;
    // fn write_u16(&mut self, addr: u64, value: u16) -> Result<(), MemoryError>;
    fn read_u32(&mut self, addr: u64) -> Result<u32, MemoryError>;
    fn write_u32(&mut self, addr: u64, value: u32) -> Result<(), MemoryError>;
    // fn read_u64(&mut self, addr: u64) -> Result<u64, MemoryError>;
    // fn write_u64(&mut self, addr: u64, value: u64) -> Result<(), MemoryError>;

    fn read_bytes(&mut self, addr: u64, data: &mut [u8]) -> Result<(), MemoryError> {
        for (i, byte) in data.iter_mut().enumerate() {
            *byte = self.read_u8(addr + i as u64)?;
        }
        Ok(())
    }

    fn write_bytes(&mut self, addr: u64, data: &[u8]) -> Result<(), MemoryError> {
        for (i, &byte) in data.iter().enumerate() {
            self.write_u8(addr + i as u64, byte)?;
        }
        Ok(())
    }
}
