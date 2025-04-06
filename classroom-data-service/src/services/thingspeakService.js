const axios = require("axios");
const ClassroomData = require("../models/ClassroomData");

class ThingSpeakService {
    constructor() {
        this.baseUrl = "https://api.thingspeak.com/channels";
        this.readApiKeys = {
            classroom1: process.env.THINGSPEAK_READ_API_KEY_CLASSROOM1,
            classroom2: process.env.THINGSPEAK_READ_API_KEY_CLASSROOM2,
        };
        this.channelIds = {
            classroom1: process.env.THINGSPEAK_CHANNEL_ID_CLASSROOM1,
            classroom2: process.env.THINGSPEAK_CHANNEL_ID_CLASSROOM2,
        };
    }

    /**
     * Fetch data from ThingSpeak for a specific classroom
     * @param {string} classroomId - The ID of the classroom (classroom1 or classroom2)
     * @param {number} results - Number of results to fetch (default: 100)
     * @returns {Promise<Array>} - Array of data points
     */
    async fetchData(classroomId, results = 100) {
        try {
            if (!this.readApiKeys[classroomId] || !this.channelIds[classroomId]) {
                throw new Error(`Missing API key or channel ID for ${classroomId}`);
            }

            const url = `${this.baseUrl}/${this.channelIds[classroomId]}/feeds.json`;
            const params = {
                api_key: this.readApiKeys[classroomId],
                results,
            };

            console.log(`Fetching data for ${classroomId}...`);
            const response = await axios.get(url, { params });

            if (!response.data || !response.data.feeds || !Array.isArray(response.data.feeds)) {
                throw new Error(`Invalid response from ThingSpeak for ${classroomId}`);
            }

            console.log(
                `Successfully fetched ${response.data.feeds.length} records for ${classroomId}`
            );
            return response.data.feeds;
        } catch (error) {
            console.error(`Error fetching data from ThingSpeak for ${classroomId}:`, error.message);
            return [];
        }
    }

    /**
     * Process and save ThingSpeak data to MongoDB
     * @param {string} classroomId - The ID of the classroom
     * @param {Array} feeds - Array of ThingSpeak feed data
     */
    async processAndSaveData(classroomId, feeds) {
        try {
            console.log(`Processing ${feeds.length} records for ${classroomId}...`);

            const operations = feeds.map((feed) => {
                const data = {
                    classroomId,
                    timestamp: new Date(feed.created_at),
                    temperature: parseFloat(feed.field1) || 0,
                    humidity: parseFloat(feed.field2) || 0,
                    co2: parseFloat(feed.field3) || 0,
                    occupancy: parseFloat(feed.field4) || 0,
                    kpiv: parseFloat(feed.field5) || 0,
                    trend: parseFloat(feed.field6) || 0,
                    alertStatus: parseInt(feed.field7) || 0,
                    modelVersion: parseFloat(feed.field8) || 1.0,
                    entryId: feed.entry_id,
                };

                // Using updateOne with upsert to avoid duplicates
                return {
                    updateOne: {
                        filter: { classroomId, entryId: feed.entry_id },
                        update: { $set: data },
                        upsert: true,
                    },
                };
            });

            if (operations.length > 0) {
                const result = await ClassroomData.bulkWrite(operations);
                console.log(
                    `Saved ${result.upsertedCount} new records and updated ${result.modifiedCount} existing records for ${classroomId}`
                );
            } else {
                console.log(`No records to save for ${classroomId}`);
            }
        } catch (error) {
            console.error(`Error processing and saving data for ${classroomId}:`, error.message);
        }
    }

    /**
     * Fetch and save data for all classrooms
     */
    async fetchAndSaveAllClassroomData() {
        try {
            console.log("Starting data fetch from ThingSpeak...");

            for (const classroomId of Object.keys(this.readApiKeys)) {
                // Avoid ThingSpeak rate limits by waiting between requests
                if (classroomId !== Object.keys(this.readApiKeys)[0]) {
                    await new Promise((resolve) => setTimeout(resolve, 15000)); // 15 seconds between requests
                }

                const feeds = await this.fetchData(classroomId);
                await this.processAndSaveData(classroomId, feeds);
            }

            console.log("Data fetch and save completed successfully");
        } catch (error) {
            console.error("Error in fetch and save operation:", error.message);
        }
    }
}

module.exports = new ThingSpeakService();
