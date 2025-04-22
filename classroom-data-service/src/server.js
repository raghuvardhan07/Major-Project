require("dotenv").config();
const express = require("express");
const path = require("path");
const fs = require("fs");
const connectDB = require("./config/database");
const certificateController = require("./controllers/certificateController");
const dataInitializationService = require("./services/dataInitializationService");
const Handlebars = require("handlebars");

// Create Express app
const app = express();
const PORT = process.env.PORT || 3000;

// Configure middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, "../public")));

// Set paths for templates and certificates
const templatesDir = path.join(__dirname, "templates");
const certificatesDir = path.join(__dirname, "../certificates");
const publicHtmlDir = path.join(__dirname, "../public");

// Ensure directories exist
if (!fs.existsSync(certificatesDir)) {
    fs.mkdirSync(certificatesDir, { recursive: true });
}
if (!fs.existsSync(publicHtmlDir)) {
    fs.mkdirSync(publicHtmlDir, { recursive: true });
}

// Copy generator HTML to public directory
const generatorTemplatePath = path.join(templatesDir, "certificate-generator.html");
const generatorPublicPath = path.join(publicHtmlDir, "certificate-generator.html");
if (fs.existsSync(generatorTemplatePath)) {
    fs.copyFileSync(generatorTemplatePath, generatorPublicPath);
}

// Format Date helper
function formatDate(date) {
    const d = new Date(date);
    return d.toLocaleDateString("en-US", {
        year: "numeric",
        month: "long",
        day: "numeric",
    });
}

// Register Handlebars helpers
Handlebars.registerHelper("toLowerCase", function (str) {
    return str.toLowerCase();
});

Handlebars.registerHelper("toFixed", function (number, decimals) {
    if (isNaN(parseFloat(number)) || !isFinite(number)) {
        return number;
    }
    return parseFloat(number).toFixed(decimals);
});

Handlebars.registerHelper("subtract", function (a, b) {
    return a - b;
});

// Routes
app.get("/", (req, res) => {
    res.redirect("/certificate-generator.html");
});

app.post("/generate-certificate", async (req, res) => {
    try {
        const { classroom, startDate, endDate } = req.body;

        if (!startDate || !endDate) {
            return res.status(400).send("Start date and end date are required");
        }

        // Parse dates
        const parsedStartDate = new Date(startDate);
        const parsedEndDate = new Date(endDate);
        // Set time to end of day for end date
        parsedEndDate.setHours(23, 59, 59, 999);

        // Ensure data exists for the requested period
        const dataCheckResult = await ensureDataForCertificate(
            classroom,
            parsedStartDate,
            parsedEndDate
        );
        if (!dataCheckResult.success) {
            return res.redirect(
                `/certificate-generator.html?error=${encodeURIComponent(dataCheckResult.message)}`
            );
        }

        // Generate certificate(s)
        let result;
        let certificateIds = [];

        if (classroom === "all") {
            // Generate for all classrooms
            result = await certificateController.generateAllCertificates(
                parsedStartDate,
                parsedEndDate
            );

            // Save each certificate
            for (const [classroomId, certificate] of Object.entries(result)) {
                if (!certificate.error) {
                    const filename = `${classroomId}_certificate_${startDate}_to_${endDate}.json`;
                    const filepath = path.join(certificatesDir, filename);
                    fs.writeFileSync(filepath, JSON.stringify(certificate, null, 2));
                    certificateIds.push({
                        classroomId,
                        filename,
                    });
                }
            }
        } else {
            // Generate for specific classroom
            result = await certificateController.generateCertificate(
                classroom,
                parsedStartDate,
                parsedEndDate
            );

            // Save certificate
            const filename = `${classroom}_certificate_${startDate}_to_${endDate}.json`;
            const filepath = path.join(certificatesDir, filename);
            fs.writeFileSync(filepath, JSON.stringify(result, null, 2));
            certificateIds.push({
                classroomId: classroom,
                filename,
            });
        }

        // Generate HTML for each certificate
        for (const certInfo of certificateIds) {
            const jsonPath = path.join(certificatesDir, certInfo.filename);
            const htmlFilename = certInfo.filename.replace(".json", ".html");
            const htmlPath = path.join(publicHtmlDir, htmlFilename);

            // Read certificate data and template
            const certificateData = JSON.parse(fs.readFileSync(jsonPath, "utf8"));
            const templateSource = fs.readFileSync(
                path.join(templatesDir, "certificate.html"),
                "utf8"
            );
            const template = Handlebars.compile(templateSource);

            // Format dates for display
            if (certificateData.period) {
                certificateData.period.startDate = formatDate(certificateData.period.startDate);
                certificateData.period.endDate = formatDate(certificateData.period.endDate);
            }
            if (certificateData.generatedAt) {
                certificateData.generatedAt = formatDate(certificateData.generatedAt);
            }

            // Render HTML and save to public directory
            const html = template(certificateData);
            fs.writeFileSync(htmlPath, html);

            // Update filename to HTML version
            certInfo.filename = htmlFilename;
        }

        // Redirect to the first certificate if available
        if (certificateIds.length > 0) {
            return res.redirect("/" + certificateIds[0].filename);
        }

        // If no certificates were generated, redirect back to generator
        return res.redirect("/certificate-generator.html?error=No+certificates+generated");
    } catch (error) {
        console.error("Error generating certificate:", error);
        res.status(500).send(`Error generating certificate: ${error.message}`);
    }
});

/**
 * Helper function to ensure data exists for certificate generation
 * @param {string} classroom - Classroom ID or 'all'
 * @param {Date} startDate - Start date
 * @param {Date} endDate - End date
 * @returns {Object} - Object with success status and message
 */
async function ensureDataForCertificate(classroom, startDate, endDate) {
    try {
        if (classroom === "all") {
            // For 'all', check each classroom
            const classroomIds = Object.keys(await getClassroomIds());
            let allSuccess = true;
            let errorMessages = [];

            for (const classroomId of classroomIds) {
                const dataExists = await dataInitializationService.hasData(
                    classroomId,
                    startDate,
                    endDate
                );

                if (!dataExists) {
                    console.log(
                        `No data found for ${classroomId} in the requested period. Fetching from ThingSpeak...`
                    );
                    const initSuccess = await dataInitializationService.initializeDataForClassroom(
                        classroomId
                    );

                    if (!initSuccess) {
                        allSuccess = false;
                        errorMessages.push(
                            `Failed to fetch data for ${classroomId} from ThingSpeak.`
                        );
                    }

                    // Check again after initialization
                    const dataExistsAfterInit = await dataInitializationService.hasData(
                        classroomId,
                        startDate,
                        endDate
                    );
                    if (!dataExistsAfterInit) {
                        allSuccess = false;
                        errorMessages.push(
                            `No data available for ${classroomId} in the requested period after initialization.`
                        );
                    }
                }
            }

            if (!allSuccess) {
                return {
                    success: false,
                    message: `Data issues: ${errorMessages.join(" ")}`,
                };
            }
        } else {
            // For a specific classroom
            const dataExists = await dataInitializationService.hasData(
                classroom,
                startDate,
                endDate
            );

            if (!dataExists) {
                console.log(
                    `No data found for ${classroom} in the requested period. Fetching from ThingSpeak...`
                );
                const initSuccess = await dataInitializationService.initializeDataForClassroom(
                    classroom
                );

                if (!initSuccess) {
                    return {
                        success: false,
                        message: `Failed to fetch data for ${classroom} from ThingSpeak.`,
                    };
                }

                // Check again after initialization
                const dataExistsAfterInit = await dataInitializationService.hasData(
                    classroom,
                    startDate,
                    endDate
                );
                if (!dataExistsAfterInit) {
                    return {
                        success: false,
                        message: `No data available for ${classroom} in the requested period after initialization.`,
                    };
                }
            }
        }

        return { success: true };
    } catch (error) {
        console.error("Error ensuring data exists:", error);
        return {
            success: false,
            message: `Error checking data availability: ${error.message}`,
        };
    }
}

/**
 * Helper function to get classroom IDs from ThingSpeak service
 */
async function getClassroomIds() {
    // This is a bit of a workaround since we don't have direct access to the ThingSpeak service
    // In a real app, you'd properly structure this to avoid circular dependencies
    try {
        const thingspeakService = require("./services/thingspeakService");
        return thingspeakService.readApiKeys;
    } catch (error) {
        console.error("Error getting classroom IDs:", error);
        return { classroom1: true, classroom2: true }; // Fallback to defaults
    }
}

// API endpoint to get recent certificates
app.get("/api/recent-certificates", (req, res) => {
    try {
        const files = fs
            .readdirSync(certificatesDir)
            .filter((file) => file.endsWith(".json"))
            .map((file) => {
                const filePath = path.join(certificatesDir, file);
                const stats = fs.statSync(filePath);
                return {
                    filename: file,
                    createdAt: stats.ctime,
                    htmlFilename: file.replace(".json", ".html"),
                };
            })
            .sort((a, b) => b.createdAt - a.createdAt) // Sort by newest first
            .slice(0, 5); // Get only the 5 most recent

        // Add additional data from each certificate
        const certificates = files.map((file) => {
            try {
                const data = JSON.parse(
                    fs.readFileSync(path.join(certificatesDir, file.filename), "utf8")
                );
                return {
                    ...file,
                    classroomId: data.classroomId,
                    grade: data.grade,
                    score: data.score,
                    period: {
                        startDate: formatDate(data.period.startDate),
                        endDate: formatDate(data.period.endDate),
                    },
                };
            } catch (err) {
                return file;
            }
        });

        res.json(certificates);
    } catch (error) {
        console.error("Error getting recent certificates:", error);
        res.status(500).json({ error: "Failed to get recent certificates" });
    }
});

// Start the server
async function startServer() {
    try {
        // Connect to MongoDB
        await connectDB();
        console.log("Connected to MongoDB");

        // Check if there's any data in the database
        const hasData = await dataInitializationService.hasAnyData();
        if (!hasData) {
            console.log("No data found in the database. Initializing from ThingSpeak...");
            await dataInitializationService.initializeAllData();
        }

        // Start listening
        app.listen(PORT, () => {
            console.log(`Server running on port ${PORT}`);
            console.log(
                `Certificate generator available at http://localhost:${PORT}/certificate-generator.html`
            );
        });
    } catch (error) {
        console.error("Failed to start server:", error);
        process.exit(1);
    }
}

// Check if being run directly or imported
if (require.main === module) {
    startServer();
}

module.exports = app;
