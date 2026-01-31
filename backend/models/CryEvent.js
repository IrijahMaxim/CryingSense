const mongoose = require('mongoose');

const cryEventSchema = new mongoose.Schema({
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
  features: {
    rmsEnergy: Number,
    zeroCrossingRate: Number,
    meanAmplitude: Number,
    spectralCentroid: Number,
    duration: Number,
    numSamples: Number
  },
  envelope: [Number],
  classification: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Classification'
  },
  processed: {
    type: Boolean,
    default: false
  }
}, {
  timestamps: true
});

// Index for querying recent events
cryEventSchema.index({ timestamp: -1 });
cryEventSchema.index({ deviceId: 1, timestamp: -1 });

module.exports = mongoose.model('CryEvent', cryEventSchema);
