import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';
import { Loader2, AlertCircle } from 'lucide-react';

const Signs = () => {
    const [classes, setClasses] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchClasses = async () => {
            try {
                const response = await axios.get('http://127.0.0.1:5000/api/classes');
                setClasses(response.data.classes);
            } catch (err) {
                console.error("Failed to fetch classes", err);
                setError("Failed to load sign classes. Ensure backend is running.");
            } finally {
                setLoading(false);
            }
        };

        fetchClasses();
    }, []);

    const containerVariants = {
        hidden: { opacity: 0 },
        visible: { opacity: 1, transition: { staggerChildren: 0.1 } }
    };

    const itemVariants = {
        hidden: { scale: 0.9, opacity: 0 },
        visible: { scale: 1, opacity: 1 }
    };

    return (
        <div className="min-h-screen py-10 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-center mb-12"
            >
                <h1 className="text-4xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-yellow-400 to-orange-500">
                    Sign Dictionary
                </h1>
                <p className="text-gray-300 max-w-3xl mx-auto mb-6">
                    Our model is currently trained to predict signs within these specific health categories.
                    <span className="italic block mt-2 text-gray-400">
                        "Till now we have made our model to predict just within these styles, but we would surely expand our model to predict other signs as well, our work is reproducible for other datasets as well."
                    </span>
                </p>
            </motion.div>

            {loading ? (
                <div className="flex justify-center items-center h-64">
                    <Loader2 className="animate-spin text-primary" size={48} />
                </div>
            ) : error ? (
                <div className="flex flex-col items-center justify-center text-red-400 h-64">
                    <AlertCircle size={48} className="mb-4" />
                    <p>{error}</p>
                </div>
            ) : (
                <motion.div
                    variants={containerVariants}
                    initial="hidden"
                    animate="visible"
                    className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8"
                >
                    {classes.map((cls) => (
                        <motion.div
                            key={cls}
                            variants={itemVariants}
                            className="bentogrid-item p-0 overflow-hidden flex flex-col"
                        >
                            <div className="aspect-video bg-black/40 relative group">
                                <video
                                    src={`http://127.0.0.1:5000/api/video_samples/${cls}`}
                                    controls
                                    loop
                                    muted
                                    playsInline
                                    className="w-full h-full object-cover"
                                >
                                    Your browser does not support the video tag.
                                </video>
                                <div className="absolute inset-0 bg-black/20 group-hover:bg-transparent transition-colors pointer-events-none" />
                            </div>
                            <div className="mt-3">
                                <h3 className="text-xl font-bold text-center capitalize text-gray-200">{cls}</h3>
                            </div>
                        </motion.div>
                    ))}
                </motion.div>
            )}
        </div>
    );
};

export default Signs;
