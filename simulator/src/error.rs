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

/// Errors that can occur while loading a binary.
#[derive(Debug)]
pub enum LoaderError {
    /// I/O error reading the file.
    Io(std::io::Error),
    /// Unsupported or malformed binary format.
    Format(String),
}

impl fmt::Display for LoaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "loader I/O error: {e}"),
            Self::Format(msg) => write!(f, "loader format error: {msg}"),
        }
    }
}

impl std::error::Error for LoaderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Format(_) => None,
        }
    }
}

impl From<std::io::Error> for LoaderError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Top-level error type for the simulator.
#[derive(Debug)]
pub enum CpuError {
    /// An instruction word could not be decoded.
    Decode(u32),
    /// A memory access faulted.
    Memory(MemoryError),
    /// A binary could not be loaded.
    Load(LoaderError),
    /// A runtime execution error.
    Execution(String),
    /// The processor has halted normally.
    Halted,
}

impl fmt::Display for CpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decode(word) => write!(f, "decode error for instruction word 0x{word:08x}"),
            Self::Memory(e) => write!(f, "memory error: {e}"),
            Self::Load(e) => write!(f, "load error: {e}"),
            Self::Execution(msg) => write!(f, "execution error: {msg}"),
            Self::Halted => write!(f, "processor halted"),
        }
    }
}

impl std::error::Error for CpuError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Memory(e) => Some(e),
            Self::Load(e) => Some(e),
            _ => None,
        }
    }
}

impl From<MemoryError> for CpuError {
    fn from(e: MemoryError) -> Self {
        Self::Memory(e)
    }
}

impl From<LoaderError> for CpuError {
    fn from(e: LoaderError) -> Self {
        Self::Load(e)
    }
}
