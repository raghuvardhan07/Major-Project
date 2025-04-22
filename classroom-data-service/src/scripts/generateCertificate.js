require("dotenv").config();
const fs = require("fs");
const path = require("path");
const connectDB = require("../config/database");
const certificateController = require("../controllers/certificateController");

// Create directory for certificates if it doesn't exist
const certificateDir = path.join(__dirname, "../../certificates");
if (!fs.existsSync(certificateDir)) {
    fs.mkdirSync(certificateDir);
}

/**
 * Generate a certificate for a specified date range and save to file
 * @param {string} classroomId - Classroom ID (classroom1 or classroom2)
 * @param {string} startDateStr - Start date in ISO format (YYYY-MM-DD)
 * @param {string} endDateStr - End date in ISO format (YYYY-MM-DD)
 */
async function generateAndSaveCertificate(classroomId, startDateStr, endDateStr) {
    try {
        console.log(
            `Generating certificate for ${classroomId} from ${startDateStr} to ${endDateStr}...`
        );

        // Parse dates
        const startDate = new Date(startDateStr);
        const endDate = new Date(endDateStr);

        // Set time to end of day for end date
        endDate.setHours(23, 59, 59, 999);

        // Generate certificate
        const certificate = await certificateController.generateCertificate(
            classroomId,
            startDate,
            endDate
        );

        // Create filename with date range
        const filename = `${classroomId}_certificate_${startDateStr}_to_${endDateStr}.json`;
        const filepath = path.join(certificateDir, filename);

        // Save certificate to file
        fs.writeFileSync(filepath, JSON.stringify(certificate, null, 2));

        console.log(`Certificate saved to ${filepath}`);
        console.log(
            `Summary: Class ${classroomId.replace("classroom", "")} - Grade: ${
                certificate.grade
            } (Score: ${certificate.score})`
        );
        console.log(`Average KPIv: ${certificate.averageMetrics.kpiv}`);
        console.log(`Recommendations:`);
        certificate.recommendations.forEach((rec, i) => {
            console.log(`${i + 1}. ${rec}`);
        });

        return certificate;
    } catch (error) {
        console.error(`Error generating certificate:`, error.message);
    }
}

/**
 * Generate certificates for all classrooms and save to files
 * @param {string} startDateStr - Start date in ISO format (YYYY-MM-DD)
 * @param {string} endDateStr - End date in ISO format (YYYY-MM-DD)
 */
async function generateAllCertificates(startDateStr, endDateStr) {
    try {
        console.log(
            `Generating certificates for all classrooms from ${startDateStr} to ${endDateStr}...`
        );

        // Parse dates
        const startDate = new Date(startDateStr);
        const endDate = new Date(endDateStr);

        // Set time to end of day for end date
        endDate.setHours(23, 59, 59, 999);

        // Generate certificates for all classrooms
        const results = await certificateController.generateAllCertificates(startDate, endDate);

        // Create filename with date range
        const filename = `all_classrooms_certificate_${startDateStr}_to_${endDateStr}.json`;
        const filepath = path.join(certificateDir, filename);

        // Save certificates to file
        fs.writeFileSync(filepath, JSON.stringify(results, null, 2));

        console.log(`Certificates saved to ${filepath}`);

        // Print summary
        console.log("\nSUMMARY:");
        for (const [classroomId, certificate] of Object.entries(results)) {
            if (certificate.error) {
                console.log(`${classroomId}: Error - ${certificate.error}`);
            } else {
                console.log(
                    `${classroomId}: Grade ${certificate.grade} (Score: ${certificate.score})`
                );
            }
        }

        return results;
    } catch (error) {
        console.error(`Error generating certificates:`, error.message);
    }
}

// Main function to run the script
async function main() {
    try {
        // Connect to DB
        await connectDB();
        console.log("Connected to MongoDB");

        // Parse command line arguments
        const args = process.argv.slice(2);
        const classroomId = args[0]; // Optional, if not provided generate for all
        const startDateStr = args[1] || "2023-01-01"; // Default to Jan 1, 2023
        const endDateStr = args[2] || new Date().toISOString().split("T")[0]; // Default to today

        if (classroomId && ["classroom1", "classroom2"].includes(classroomId)) {
            // Generate certificate for specific classroom
            await generateAndSaveCertificate(classroomId, startDateStr, endDateStr);
        } else {
            // Generate certificates for all classrooms
            await generateAllCertificates(startDateStr, endDateStr);
        }

        console.log("Certificate generation complete");
        process.exit(0);
    } catch (error) {
        console.error("Error:", error.message);
        process.exit(1);
    }
}

// Run the script
main();
