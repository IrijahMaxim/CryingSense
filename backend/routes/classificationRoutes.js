const express = require('express');
const router = express.Router();
const classificationController = require('../controllers/classificationController');

// POST /api/classifications - Create new classification
router.post('/', classificationController.createClassification);

// GET /api/classifications - Get all classifications
router.get('/', classificationController.getAllClassifications);

// GET /api/classifications/stats - Get statistics
router.get('/stats', classificationController.getClassificationStats);

// GET /api/classifications/:id - Get classification by ID
router.get('/:id', classificationController.getClassificationById);

module.exports = router;
