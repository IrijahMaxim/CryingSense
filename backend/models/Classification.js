const mongoose = require('mongoose');

const classificationSchema = new mongoose.Schema({
  cryEvent: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'CryEvent',
    required: true
  },
  deviceId: {
    type: String,
    required: true,
    index: true
  },
  timestamp: {
    type: Date,
    default: Date.now,
    required: true,
    index: true
  },
  predictedClass: {
    type: String,
    required: true,
    enum: ['belly_pain', 'burp', 'discomfort', 'hunger', 'tired']
  },
  confidence: {
    type: Number,
    required: true,
    min: 0,
    max: 1
  },
  probabilities: {
    belly_pain: Number,
    burp: Number,
    discomfort: Number,
    hunger: Number,
    tired: Number
  },
  inferenceTimeMs: Number,
  notificationSent: {
    type: Boolean,
    default: false
  }
}, {
  timestamps: true
});

// Index for analytics queries
classificationSchema.index({ predictedClass: 1, timestamp: -1 });
classificationSchema.index({ deviceId: 1, timestamp: -1 });

module.exports = mongoose.model('Classification', classificationSchema);
