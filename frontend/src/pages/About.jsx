import React from 'react';
import { motion } from 'framer-motion';
import { Linkedin, Code2, Database, Box, Server, Cpu } from 'lucide-react';

const About = () => {
    return (
        <div className="min-h-screen py-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-center mb-16"
            >
                <h1 className="text-4xl md:text-5xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-pink-400">About Our Project</h1>
                <p className="text-gray-400 max-w-2xl mx-auto">Bridging communication gaps with cutting-edge AI.</p>
            </motion.div>

            {/* Bentogrid Section */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-16">
                {/* Main Description - Spans 2 cols */}
                <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.1 }}
                    className="lg:col-span-2 bentogrid-item flex flex-col justify-center"
                >
                    <h2 className="text-2xl font-bold mb-4 text-blue-400">Our Mission</h2>
                    <p className="text-gray-300 leading-relaxed mb-4">
                        This model is specifically designed to detect health-related sign language classes.
                        It helps mute individuals or those with speech impairments to express their health conditions to doctors or caregivers effectively through webcam solutions.
                        Our goal is to assist in accurate and timely medical diagnoses by removing the communication barrier.
                    </p>
                    <p className="text-gray-400 text-sm italic">
                        Leveraging the power of MediaPipe and deep learning to interpret gestures in real-time.
                    </p>
                </motion.div>

                {/* Features Mini-list */}
                <motion.div
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.2 }}
                    className="bentogrid-item"
                >
                    <h3 className="text-xl font-semibold mb-4 text-purple-400">Key Capabilities</h3>
                    <ul className="space-y-3 text-gray-300">
                        <li className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-blue-500" />Real-time Inference</li>
                        <li className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-purple-500" />High Accuracy</li>
                        <li className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-pink-500" />Privacy Focused</li>
                        <li className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-cyan-500" />Works Offline</li>
                    </ul>
                </motion.div>
            </div>

            {/* Tech Stack */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                className="mb-16"
            >
                <h2 className="text-3xl font-bold mb-8 text-center bg-clip-text text-transparent bg-gradient-to-r from-green-400 to-emerald-600">Technology Stack</h2>
                <div className="grid grid-cols-2 md:grid-cols-5 gap-6">
                    {[
                        { name: 'Flask', icon: <Server /> },
                        { name: 'React', icon: <Code2 /> },
                        { name: 'TensorFlow', icon: <Box /> },
                        { name: 'MediaPipe', icon: <Cpu /> },
                        { name: 'OpenCV', icon: <Database /> }
                    ].map((tech, i) => (
                        <div key={i} className="bentogrid-item flex flex-col items-center justify-center p-4">
                            <div className="text-emerald-400 mb-2">{tech.icon}</div>
                            <span className="font-medium text-gray-300">{tech.name}</span>
                        </div>
                    ))}
                </div>
            </motion.div>

            {/* Team Section */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
            >
                <h2 className="text-3xl font-bold mb-8 text-center bg-clip-text text-transparent bg-gradient-to-r from-orange-400 to-red-500">Our Team</h2>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                    {[
                        { name: 'Shinjan Saha', role: 'ML Engineer', linkedin: "https://www.linkedin.com/in/shinjan-saha-1bb744319/" },
                        { name: 'Satyabrata Das Adhikari', role: 'Full Stack Developer', linkedin: "https://www.linkedin.com/in/satyabrata-das-adhikari-1813a7324/" },
                        { name: 'Sayan Sk', role: 'Frontend Architect', linkedin: "https://www.linkedin.com/in/sayan-sk-092203318/" }
                    ].map((member, i) => (
                        <div key={i} className="bentogrid-item text-center">
                            <div className="w-20 h-20 rounded-full bg-gray-700 mx-auto mb-4 flex items-center justify-center text-2xl font-bold text-gray-400">
                                {member.name[0]}
                            </div>
                            <h3 className="text-xl font-bold text-white mb-1">{member.name}</h3>
                            <p className="text-gray-400 text-sm mb-4">{member.role}</p>
                            <a href="#" className="inline-flex items-center gap-2 text-blue-400 hover:text-blue-300 transition-colors">
                                <Linkedin size={20} />
                                <span>Connect</span>
                            </a>
                        </div>
                    ))}
                </div>
            </motion.div>
        </div>
    );
};

export default About;
