const Device = require('../models/Device');

// Register device
exports.registerDevice = async (req, res) => {
  try {
    const { deviceId, name, type, location, firmwareVersion } = req.body;
    
    if (!deviceId || !name || !type) {
      return res.status(400).json({ error: 'Device ID, name, and type are required' });
    }
    
    // Check if device already exists
    let device = await Device.findOne({ deviceId });
    
    if (device) {
      // Update existing device
      device.name = name;
      device.type = type;
      device.location = location;
      device.firmwareVersion = firmwareVersion;
      device.status = 'online';
      device.lastSeen = Date.now();
      await device.save();
      
      return res.json({
        success: true,
        message: 'Device updated',
        data: device
      });
    }
    
    // Create new device
    device = new Device({
      deviceId,
      name,
      type,
      location,
      firmwareVersion,
      status: 'online',
      lastSeen: Date.now()
    });
    
    await device.save();
    
    res.status(201).json({
      success: true,
      message: 'Device registered',
      data: device
    });
  } catch (error) {
    console.error('Error registering device:', error);
    res.status(500).json({ error: error.message });
  }
};

// Update device status
exports.updateDeviceStatus = async (req, res) => {
  try {
    const { deviceId } = req.params;
    const { status } = req.body;
    
    const device = await Device.findOneAndUpdate(
      { deviceId },
      { status, lastSeen: Date.now() },
      { new: true }
    );
    
    if (!device) {
      return res.status(404).json({ error: 'Device not found' });
    }
    
    res.json({
      success: true,
      message: 'Device status updated',
      data: device
    });
  } catch (error) {
    console.error('Error updating device status:', error);
    res.status(500).json({ error: error.message });
  }
};

// Get all devices
exports.getAllDevices = async (req, res) => {
  try {
    const devices = await Device.find().sort({ lastSeen: -1 });
    
    res.json({
      success: true,
      data: devices
    });
  } catch (error) {
    console.error('Error fetching devices:', error);
    res.status(500).json({ error: error.message });
  }
};

// Get device by ID
exports.getDeviceById = async (req, res) => {
  try {
    const device = await Device.findOne({ deviceId: req.params.deviceId });
    
    if (!device) {
      return res.status(404).json({ error: 'Device not found' });
    }
    
    res.json({ success: true, data: device });
  } catch (error) {
    console.error('Error fetching device:', error);
    res.status(500).json({ error: error.message });
  }
};

module.exports = exports;
