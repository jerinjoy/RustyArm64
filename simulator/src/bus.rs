use std::ops::RangeInclusive;

use crate::io_device::IoDevice;

#[derive(Default)]
pub struct Bus {
    devices: Vec<(RangeInclusive<u64>, Box<dyn IoDevice>)>,
}

impl Bus {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, range: RangeInclusive<u64>, device: Box<dyn IoDevice>) {
        self.devices.push((range, device));
    }

    fn find_device_mut(&mut self, address: u64) -> Option<&mut dyn IoDevice> {
        for (range, device) in &mut self.devices {
            if range.contains(&address) {
                return Some(device.as_mut());
            }
        }
        None
    }
}
