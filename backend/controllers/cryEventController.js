const CryEvent = require('../models/CryEvent');

// Create a new cry event
exports.createCryEvent = async (req, res) => {
  try {
    const { deviceId, timestamp, features, envelope } = req.body;
    
    // Validate required fields
    if (!deviceId) {
      return res.status(400).json({ error: 'Device ID is required' });
    }
    
    // Create cry event
    const cryEvent = new CryEvent({
      deviceId,
      timestamp: timestamp || Date.now(),
      features: features ? {
        rmsEnergy: features.rms_energy || features.rmsEnergy,
        zeroCrossingRate: features.zcr || features.zero_crossing_rate || features.zeroCrossingRate,
        meanAmplitude: features.mean_amplitude || features.meanAmplitude,
        spectralCentroid: features.spectral_centroid || features.spectralCentroid,
        duration: features.duration,
        numSamples: features.num_samples || features.numSamples
      } : {},
      envelope: envelope || []
    });
    
    await cryEvent.save();
    
    res.status(201).json({
      success: true,
      message: 'Cry event created',
      data: cryEvent
    });
  } catch (error) {
    console.error('Error creating cry event:', error);
    res.status(500).json({ error: error.message });
  }
};

// Get all cry events
exports.getAllCryEvents = async (req, res) => {
  try {
    const { deviceId, startDate, endDate, limit = 50, page = 1 } = req.query;
    
    const query = {};
    if (deviceId) query.deviceId = deviceId;
    if (startDate || endDate) {
      query.timestamp = {};
      if (startDate) query.timestamp.$gte = new Date(startDate);
      if (endDate) query.timestamp.$lte = new Date(endDate);
    }
    
    const skip = (page - 1) * limit;
    
    const events = await CryEvent.find(query)
      .sort({ timestamp: -1 })
      .limit(parseInt(limit))
      .skip(skip)
      .populate('classification');
    
    const total = await CryEvent.countDocuments(query);
    
    res.json({
      success: true,
      data: events,
      pagination: {
        total,
        page: parseInt(page),
        pages: Math.ceil(total / limit),
        limit: parseInt(limit)
      }
    });
  } catch (error) {
    console.error('Error fetching cry events:', error);
    res.status(500).json({ error: error.message });
  }
};

// Get cry event by ID
exports.getCryEventById = async (req, res) => {
  try {
    const event = await CryEvent.findById(req.params.id).populate('classification');
    
    if (!event) {
      return res.status(404).json({ error: 'Cry event not found' });
    }
    
    res.json({ success: true, data: event });
  } catch (error) {
    console.error('Error fetching cry event:', error);
    res.status(500).json({ error: error.message });
  }
};

// Delete cry event
exports.deleteCryEvent = async (req, res) => {
  try {
    const event = await CryEvent.findByIdAndDelete(req.params.id);
    
    if (!event) {
      return res.status(404).json({ error: 'Cry event not found' });
    }
    
    res.json({ success: true, message: 'Cry event deleted' });
  } catch (error) {
    console.error('Error deleting cry event:', error);
    res.status(500).json({ error: error.message });
  }
};

// Get cry event statistics
exports.getCryEventStats = async (req, res) => {
  try {
    const { deviceId, days = 7 } = req.query;
    
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - parseInt(days));
    
    const query = { timestamp: { $gte: startDate } };
    if (deviceId) query.deviceId = deviceId;
    
    const total = await CryEvent.countDocuments(query);
    const processed = await CryEvent.countDocuments({ ...query, processed: true });
    const unprocessed = total - processed;
    
    res.json({
      success: true,
      data: {
        total,
        processed,
        unprocessed,
        period: `Last ${days} days`
      }
    });
  } catch (error) {
    console.error('Error fetching stats:', error);
    res.status(500).json({ error: error.message });
  }
};

module.exports = exports;
