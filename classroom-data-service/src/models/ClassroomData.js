const mongoose = require("mongoose");

const ClassroomDataSchema = new mongoose.Schema({
    classroomId: {
        type: String,
        required: true,
        enum: ["classroom1", "classroom2"],
        index: true,
    },
    timestamp: {
        type: Date,
        required: true,
        index: true,
    },
    temperature: {
        type: Number,
        required: true,
    },
    humidity: {
        type: Number,
        required: true,
    },
    co2: {
        type: Number,
        required: true,
    },
    occupancy: {
        type: Number,
        required: true,
    },
    kpiv: {
        type: Number,
        required: true,
    },
    trend: {
        type: Number,
        default: 0,
    },
    alertStatus: {
        type: Number,
        default: 0,
    },
    modelVersion: {
        type: Number,
        default: 1.0,
    },
    entryId: {
        type: Number,
        required: true,
        index: true,
    },
});

// Compound index for faster lookups
ClassroomDataSchema.index({ classroomId: 1, timestamp: 1 }, { unique: true });

module.exports = mongoose.model("ClassroomData", ClassroomDataSchema);
