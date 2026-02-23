import React from 'react';
import { motion } from 'framer-motion';
import { Camera, BarChart2, Eye, Crosshair, Zap, Activity } from 'lucide-react';

const Home = () => {
    const containerVariants = {
        hidden: { opacity: 0 },
        visible: {
            opacity: 1,
            transition: { staggerChildren: 0.1 }
        }
    };

    const itemVariants = {
        hidden: { y: 20, opacity: 0 },
        visible: { y: 0, opacity: 1 }
    };

    const features = [
        { icon: <Camera size={32} />, title: 'Live Detection', desc: 'Real-time sign language recognition using advanced computer vision.' },
        { icon: <BarChart2 size={32} />, title: 'Smart Analytics', desc: 'Detailed confidence scores and detection history analytics.' },
        { icon: <Eye size={32} />, title: 'Visual Overlay', desc: 'Intuitive augmented reality overlay on your camera feed.' },
        { icon: <Crosshair size={32} />, title: 'Precision Mode', desc: 'Enhanced accuracy for medical and health-related signs.' },
        { icon: <Zap size={32} />, title: 'Instant Feedback', desc: 'Zero-latency feedback for seamless communication.' },
        { icon: <Activity size={32} />, title: 'Health Focus', desc: 'Specialized in detecting health conditions and symptoms.' },
    ];

    return (
        <div className="min-h-full">
            {/* Hero Section */}
            <section className="relative h-[90vh] flex items-center justify-center overflow-hidden">
                {/* Background Elements */}
                <div className="absolute inset-0 bg-background">
                    <div className="absolute top-0 -left-1/4 w-1/2 h-full bg-blue-600/20 blur-[150px] rounded-full mix-blend-screen" />
                    <div className="absolute bottom-0 -right-1/4 w-1/2 h-full bg-purple-600/20 blur-[150px] rounded-full mix-blend-screen" />
                </div>

                <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
                    <motion.h1
                        initial={{ scale: 0.8, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        transition={{ duration: 0.8, ease: "easeOut" }}
                        className="text-6xl md:text-8xl font-black tracking-tight mb-8"
                    >
                        <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-400 via-sky-300 to-purple-400">
                            SignAI
                        </span>
                    </motion.h1>

                    <motion.p
                        initial={{ y: 20, opacity: 0 }}
                        animate={{ y: 0, opacity: 1 }}
                        transition={{ delay: 0.3, duration: 0.8 }}
                        className="text-xl md:text-2xl text-gray-300 mb-12 max-w-3xl mx-auto leading-relaxed"
                    >
                        Empowering communication through AI. Bridging the gap for those who cannot express their health conditions verbally.
                    </motion.p>

                    <motion.div
                        initial={{ y: 20, opacity: 0 }}
                        animate={{ y: 0, opacity: 1 }}
                        transition={{ delay: 0.6 }}
                        className="flex flex-col sm:flex-row gap-4 justify-center"
                    >
                        <a href="/detection" className="px-8 py-4 bg-primary text-primary-foreground rounded-full text-lg font-bold hover:bg-blue-600 transition-all shadow-lg shadow-blue-500/25">
                            Start Detection
                        </a>
                        <a href="/about" className="px-8 py-4 bg-white/10 text-white rounded-full text-lg font-medium hover:bg-white/20 transition-all backdrop-blur-sm">
                            Learn More
                        </a>
                    </motion.div>
                </div>
            </section>

            {/* Features Section */}
            <section className="py-24 bg-background/50 relative">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="text-center mb-16">
                        <h2 className="text-3xl md:text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-400 mb-4">
                            Why Choose SignAI?
                        </h2>
                        <p className="text-gray-400">Advanced features designed for reliability and ease of use.</p>
                    </div>

                    <motion.div
                        variants={containerVariants}
                        initial="hidden"
                        whileInView="visible"
                        viewport={{ once: true }}
                        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8"
                    >
                        {features.map((feature, idx) => (
                            <motion.div
                                key={idx}
                                variants={itemVariants}
                                className="bentogrid-item flex flex-col items-center text-center group"
                            >
                                <div className="p-4 rounded-full bg-blue-500/10 text-blue-400 mb-6 group-hover:scale-110 transition-transform duration-300">
                                    {feature.icon}
                                </div>
                                <h3 className="text-xl font-semibold mb-3">{feature.title}</h3>
                                <p className="text-gray-400 text-sm leading-relaxed">{feature.desc}</p>
                            </motion.div>
                        ))}
                    </motion.div>
                </div>
            </section>
        </div>
    );
};

export default Home;
