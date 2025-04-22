const ClassroomData = require("../models/ClassroomData");

class CertificateService {
    /**
     * Generate a certificate score using a sigmoid function based on KPIv values
     * @param {number} kpiv - The KPIv value
     * @returns {number} - Score between 0 and 100
     */
    calculateScore(kpiv) {
        // Sigmoid function to transform KPIv (typically 0-2) to a score (0-100)
        // Lower KPIv is better, so we invert it: 2-kpiv
        // A KPIv of 0 should give a high score, KPIv of 2 should give a low score
        const normalizedKpiv = Math.max(0, Math.min(2, kpiv)); // Clamp between 0 and 2
        const invertedKpiv = 2 - normalizedKpiv; // Invert so lower KPIv means higher score

        // Sigmoid function: 100 * (1 / (1 + e^(-k * (x - midpoint))))
        // where k controls steepness, midpoint is the center point
        const k = 3; // Steepness factor
        const midpoint = 1; // Center point (KPIv of 1 gives score of 50)

        const score = 100 * (1 / (1 + Math.exp(-k * (invertedKpiv - midpoint))));
        return Math.round(score);
    }

    /**
     * Get a grade based on the score
     * @param {number} score - The score (0-100)
     * @returns {string} - The grade (A+, A, B, C, D, F)
     */
    getGrade(score) {
        if (score >= 95) return "A+";
        if (score >= 90) return "A";
        if (score >= 80) return "B";
        if (score >= 70) return "C";
        if (score >= 60) return "D";
        return "F";
    }

    /**
     * Get recommendations based on the score
     * @param {number} score - The score (0-100)
     * @param {Object} metrics - The classroom metrics
     * @returns {Array} - Array of recommendation strings
     */
    getRecommendations(score, metrics) {
        const recommendations = [];

        if (metrics.co2 > 800) {
            recommendations.push("Improve ventilation to reduce CO2 levels.");
        }

        if (metrics.temperature < 20 || metrics.temperature > 26) {
            recommendations.push(
                `Adjust temperature to optimal range (20-26°C). Current: ${metrics.temperature.toFixed(
                    1
                )}°C.`
            );
        }

        if (metrics.humidity < 40 || metrics.humidity > 60) {
            recommendations.push(
                `Maintain humidity between 40-60%. Current: ${metrics.humidity.toFixed(1)}%.`
            );
        }

        if (metrics.occupancy > metrics.capacity * 0.8) {
            recommendations.push(
                `Consider redistributing occupants. Current occupancy: ${Math.round(
                    metrics.occupancy
                )}/${metrics.capacity}.`
            );
        }

        if (recommendations.length === 0 && score > 80) {
            recommendations.push("Maintain current environmental conditions.");
        }

        return recommendations;
    }

    /**
     * Generate a certificate for a specified classroom over a date range
     * @param {string} classroomId - The classroom ID
     * @param {Date} startDate - Start date for the certificate period
     * @param {Date} endDate - End date for the certificate period
     * @returns {Object} - Certificate data
     */
    async generateCertificate(classroomId, startDate, endDate) {
        try {
            // Get classroom data for the specified period
            const data = await ClassroomData.find({
                classroomId,
                timestamp: { $gte: startDate, $lte: endDate },
            }).sort({ timestamp: 1 });

            if (data.length === 0) {
                throw new Error(
                    `No data found for ${classroomId} between ${startDate} and ${endDate}`
                );
            }

            // Calculate average metrics
            const metrics = data.reduce(
                (acc, record) => {
                    acc.temperature += record.temperature;
                    acc.humidity += record.humidity;
                    acc.co2 += record.co2;
                    acc.occupancy += record.occupancy;
                    acc.kpiv += record.kpiv;
                    return acc;
                },
                {
                    temperature: 0,
                    humidity: 0,
                    co2: 0,
                    occupancy: 0,
                    kpiv: 0,
                }
            );

            const count = data.length;
            metrics.temperature /= count;
            metrics.humidity /= count;
            metrics.co2 /= count;
            metrics.occupancy /= count;
            metrics.kpiv /= count;

            // Add capacity (estimated from the classroom name - can be replaced with actual data)
            metrics.capacity = classroomId === "classroom1" ? 25 : 30;

            // Get min/max values for each metric
            const minMax = data.reduce(
                (acc, record) => {
                    // Min values
                    acc.minTemperature = Math.min(acc.minTemperature, record.temperature);
                    acc.minHumidity = Math.min(acc.minHumidity, record.humidity);
                    acc.minCO2 = Math.min(acc.minCO2, record.co2);
                    acc.minKpiv = Math.min(acc.minKpiv, record.kpiv);

                    // Max values
                    acc.maxTemperature = Math.max(acc.maxTemperature, record.temperature);
                    acc.maxHumidity = Math.max(acc.maxHumidity, record.humidity);
                    acc.maxCO2 = Math.max(acc.maxCO2, record.co2);
                    acc.maxKpiv = Math.max(acc.maxKpiv, record.kpiv);

                    return acc;
                },
                {
                    minTemperature: Infinity,
                    minHumidity: Infinity,
                    minCO2: Infinity,
                    minKpiv: Infinity,
                    maxTemperature: -Infinity,
                    maxHumidity: -Infinity,
                    maxCO2: -Infinity,
                    maxKpiv: -Infinity,
                }
            );

            // Calculate score based on average KPIv
            const score = this.calculateScore(metrics.kpiv);
            const grade = this.getGrade(score);
            const recommendations = this.getRecommendations(score, metrics);

            // Generate certificate
            return {
                classroomId,
                period: {
                    startDate,
                    endDate,
                },
                averageMetrics: {
                    temperature: metrics.temperature.toFixed(1),
                    humidity: metrics.humidity.toFixed(1),
                    co2: Math.round(metrics.co2),
                    occupancy: metrics.occupancy.toFixed(1),
                    kpiv: metrics.kpiv.toFixed(2),
                },
                minMax,
                score,
                grade,
                recommendations,
                dataPoints: count,
                generatedAt: new Date(),
            };
        } catch (error) {
            console.error(`Error generating certificate for ${classroomId}:`, error.message);
            throw error;
        }
    }
}

module.exports = new CertificateService();
