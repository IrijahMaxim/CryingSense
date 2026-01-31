const express = require('express');
const router = express.Router();
const cryEventController = require('../controllers/cryEventController');

// POST /api/cry-events - Create new cry event
router.post('/', cryEventController.createCryEvent);

// GET /api/cry-events - Get all cry events
router.get('/', cryEventController.getAllCryEvents);

// GET /api/cry-events/stats - Get statistics
router.get('/stats', cryEventController.getCryEventStats);

// GET /api/cry-events/:id - Get cry event by ID
router.get('/:id', cryEventController.getCryEventById);

// DELETE /api/cry-events/:id - Delete cry event
router.delete('/:id', cryEventController.deleteCryEvent);

module.exports = router;
