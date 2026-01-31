const Classification = require('../models/Classification');
const CryEvent = require('../models/CryEvent');

// Create classification
exports.createClassification = async (req, res) => {
  try {
    const {
      cryEventId,
      deviceId,
      timestamp,
      predicted_class,
      predictedClass,
      confidence,
      probabilities,
      inference_time_ms,
      inferenceTimeMs
    } = req.body;
    
    // Validate required fields
    if (!deviceId || !(predicted_class || predictedClass) || confidence === undefined) {
      return res.status(400).json({ error: 'Missing required fields' });
    }
    
    // Create classification
    const classification = new Classification({
      cryEvent: cryEventId,
      deviceId,
      timestamp: timestamp || Date.now(),
      predictedClass: predicted_class || predictedClass,
      confidence,
      probabilities,
      inferenceTimeMs: inference_time_ms || inferenceTimeMs
    });
    
    await classification.save();
    
    // Update cry event if provided
    if (cryEventId) {
      await CryEvent.findByIdAndUpdate(cryEventId, {
        classification: classification._id,
        processed: true
      });
    }
    
    res.status(201).json({
      success: true,
      message: 'Classification created',
      data: classification
    });
  } catch (error) {
    console.error('Error creating classification:', error);
    res.status(500).json({ error: error.message });
  }
};

// Get all classifications
exports.getAllClassifications = async (req, res) => {
  try {
    const { deviceId, predictedClass, startDate, endDate, limit = 50, page = 1 } = req.query;
    
    const query = {};
    if (deviceId) query.deviceId = deviceId;
    if (predictedClass) query.predictedClass = predictedClass;
    if (startDate || endDate) {
      query.timestamp = {};
      if (startDate) query.timestamp.$gte = new Date(startDate);
      if (endDate) query.timestamp.$lte = new Date(endDate);
    }
    
    const skip = (page - 1) * limit;
    
    const classifications = await Classification.find(query)
      .sort({ timestamp: -1 })
      .limit(parseInt(limit))
      .skip(skip)
      .populate('cryEvent');
    
    const total = await Classification.countDocuments(query);
    
    res.json({
      success: true,
      data: classifications,
      pagination: {
        total,
        page: parseInt(page),
        pages: Math.ceil(total / limit),
        limit: parseInt(limit)
      }
    });
  } catch (error) {
    console.error('Error fetching classifications:', error);
    res.status(500).json({ error: error.message });
  }
};

// Get classification by ID
exports.getClassificationById = async (req, res) => {
  try {
    const classification = await Classification.findById(req.params.id).populate('cryEvent');
    
    if (!classification) {
      return res.status(404).json({ error: 'Classification not found' });
    }
    
    res.json({ success: true, data: classification });
  } catch (error) {
    console.error('Error fetching classification:', error);
    res.status(500).json({ error: error.message });
  }
};

// Get classification statistics
exports.getClassificationStats = async (req, res) => {
  try {
    const { deviceId, days = 7 } = req.query;
    
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - parseInt(days));
    
    const query = { timestamp: { $gte: startDate } };
    if (deviceId) query.deviceId = deviceId;
    
    // Aggregate by predicted class
    const classDistribution = await Classification.aggregate([
      { $match: query },
      { $group: { _id: '$predictedClass', count: { $sum: 1 } } },
      { $sort: { count: -1 } }
    ]);
    
    // Average confidence
    const avgConfidence = await Classification.aggregate([
      { $match: query },
      { $group: { _id: null, avgConfidence: { $avg: '$confidence' } } }
    ]);
    
    res.json({
      success: true,
      data: {
        classDistribution,
        averageConfidence: avgConfidence[0]?.avgConfidence || 0,
        period: `Last ${days} days`
      }
    });
  } catch (error) {
    console.error('Error fetching stats:', error);
    res.status(500).json({ error: error.message });
  }
};

module.exports = exports;
