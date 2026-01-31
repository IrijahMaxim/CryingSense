const express = require('express');
const router = express.Router();
const deviceController = require('../controllers/deviceController');

// POST /api/devices - Register device
router.post('/', deviceController.registerDevice);

// GET /api/devices - Get all devices
router.get('/', deviceController.getAllDevices);

// GET /api/devices/:deviceId - Get device by ID
router.get('/:deviceId', deviceController.getDeviceById);

// PUT /api/devices/:deviceId/status - Update device status
router.put('/:deviceId/status', deviceController.updateDeviceStatus);

module.exports = router;
