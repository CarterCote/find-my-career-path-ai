'use client';
import { useState, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Image from "next/image";
import { Button } from "@/components/ui/button";
import EmailCapture from "@/components/EmailCapture";
import Header from "@/components/Header";
import SkillsGrid from "@/components/SkillGrid";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

import { Textarea } from "@/components/ui/textarea";


export default function Try() {
    const [currentStep, setCurrentStep] = useState(0);
    const [dialogOpen, setDialogOpen] = useState(false);
    const [howItWorksDialogOpen, setHowItWorksDialogOpen] = useState(false);
    const [skills, setSkills] = useState([
        "build", "design/create", "customer/client focus", "persuade/sell", 
        "programming", "solve complex problems", "think critically", 
        "technical expertise", "strategic planning", "adaptability",
        "make decisions", "work in teams", "lead", "manage and develop others",
        "build relationships", "global mindset", "storytelling", "empathy",
        "interpret data", "quantitative", "communicate verbally", 
        "project management", "develop curriculum", "analyze", "research",
        "write", "teach", "social media", "make presentations", 
        "organize", "manage finances"
    ].filter((value, index, self) => self.indexOf(value) === index));

    const [workCulture, setWorkCulture] = useState([
        "inspiring", "fast-paced", "growth potential", "ethical leadership",
        "collaboration", "entrepreneurial", "innovation", "professional development",
        "engaging work", "geographical location", "flexibility", "service focus",
        "respect", "workplace surroundings", "fairness", "supportive environment",
        "transparency", "mentoring", "variety", "travel", "challenging",
        "predictability", "humor", "inclusive workplace", "salary", "fun",
        "stability", "recognition", "structure", "life balance", "competition"
    ].filter((value, index, self) => self.indexOf(value) === index));

    const [coreValues, setCoreValues] = useState([
        "integrity", "personal development", "family/friends", "fulfilling one's potential",
        "authenticity", "impact", "creativity", "excellence", "intellectual challenge",
        "success", "achievement", "community", "gratitude", "learning", "social justice",
        "autonomy", "compassion", "relationships", "safety", "diversity", "joy",
        "belonging", "environmental responsibility", "adventure", "helping", "prestige",
        "harmony", "spirituality", "trust", "well-being", "simplicity"
    ].filter((value, index, self) => self.indexOf(value) === index));

    const [highSkills, setHighSkills] = useState<string[]>([]);
    const [topFiveSkills, setTopFiveSkills] = useState<string[]>([]);
    const [step3HighSkills, setStep3HighSkills] = useState<string[]>([]);
    const getHighPrioritySkills = () => {
        return step3HighSkills;
    };

    const handleNext = () => {
        if (currentStep === 3) {
            console.log('Skills after step 3:', JSON.stringify(topSkills, null, 2));
        }
        if (currentStep === 4) {
            console.log('Work Culture after step 4:', JSON.stringify(topWorkCulture, null, 2));
        }
        if (currentStep === 5) {
            console.log('Core Values after step 5:', JSON.stringify(topCoreValues, null, 2));
            const sortedData = {
                skills: topSkills,
                work_culture: topWorkCulture,
                core_values: topCoreValues
            };
            console.log('Complete Sorted Data:', JSON.stringify(sortedData, null, 2));
        }
        if (currentStep === 2) {
            setDialogOpen(true);
        }
        if (currentStep === 3) {
            const highPrioritySkills = getHighPrioritySkills();
            setHighSkills(highPrioritySkills);
        }
        setCurrentStep(prev => prev + 1);
    };

    const handleBack = () => {
        setCurrentStep(prev => prev - 1);
    };

    const fadeVariants = {
        initial: { opacity: 0 },
        animate: { opacity: 1 },
        exit: { opacity: 0 }
    };

    // Add state for tracking top items from each grid
    const [topSkills, setTopSkills] = useState<string[]>([]);
    const [topWorkCulture, setTopWorkCulture] = useState<string[]>([]);
    const [topCoreValues, setTopCoreValues] = useState<string[]>([]);
    const [additionalInterests, setAdditionalInterests] = useState('');

    // Add function to send data to backend
    const handleFinish = async () => {
        try {
            const dataToSend = {
                skills: topSkills,
                work_culture: topWorkCulture,  // match Python backend naming
                core_values: topCoreValues,    // match Python backend naming
                additional_interests: additionalInterests
            };

            console.log('Sending data:', dataToSend);

            const response = await fetch('/api/profile', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(dataToSend)
            });

            const responseText = await response.text();
            console.log('Raw response:', responseText);

            if (!response.ok) {
                throw new Error(`Failed to save profile: ${responseText}`);
            }

            const result = responseText ? JSON.parse(responseText) : {};
            console.log('Profile saved successfully:', result);
            
        } catch (error) {
            console.error('Error saving profile:', error);
        }
    };

  return (
    <div className="min-h-screen p-8 pb-20 gap-16 sm:p-2">
      <Header />
      <main className="flex items-center justify-center min-h-[82vh]">
        <AnimatePresence mode="wait">
          {currentStep === 0 && (
            <motion.div
              key="step0"
              className="w-1/2 flex flex-col justify-center space-y-6 items-start py-24"
              variants={fadeVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              transition={{ duration: 0.2 }}
            >
              <div className="flex flex-row space-x-2 items-center">
                <div className="flex flex-row space-x-2 bg-[#f28d3533] px-3 py-1 rounded-full">
                  <p className="font-semibold text-sm text-[#f28d35]">Alpha</p>
                </div>
                <p className="text-[16px] font-bold">This system is currently in an alpha stage.</p>
                <a href="mailto:chcote@stanford.edu" className="underline font-medium">Share your feedback here.</a>
              </div>
              <p className="font-circular font-bold text-4xl tracking-tight">Take the Pathways assessment</p>
              <div className="flex flex-col space-y-4">
                <p className="">We'll make connections between the skills you enjoy using, work environments where you thrive, and values that drive you. Then, our AI system will match you with personalized career pathways, and outline the steps to help you get there.</p>
              </div>
              <div className="flex flex-row items-center space-x-3">
                <Button variant="secondary" onClick={handleNext}>Take the survey</Button>
                <Button className="bg-[#0C3C60] rounded-xl py-3" onClick={() => setHowItWorksDialogOpen(true)}>How it works</Button>
              </div>
              <Dialog open={howItWorksDialogOpen} onOpenChange={setHowItWorksDialogOpen}>
                <DialogContent>
                    <DialogHeader>
                        <DialogTitle className="font-circular font-bold text-3xl tracking-tight">How this works</DialogTitle>
                    </DialogHeader>
                    <DialogDescription>
                        <div className="flex flex-col space-y-4 leading-[180%]">
                            <p className="">Using AI, Pathways transforms a user's skills, values, and interests into tailored career pathway recommendations, drawing on real-world examples from individuals with similar backgrounds who are actively pursuing those paths.</p>
                            <p>The system then fine-tunes these recommendations to reflect the user's unique preferences, prioritizing the most relevant options.</p>
                            <p>As users engage with the suggestions and provide feedback, Pathways adapts by learning from this input, continuously refining its recommendations to support the user's career journey as it evolves.</p>
                            <p>You will begin with an onboarding assessment that determines the skills, work culture, and core values that are important to you in your ideal career path.</p>
                        </div>
                    </DialogDescription>
                    <Button className="mt-4" variant="secondary" onClick={() => setHowItWorksDialogOpen(false)}>Close</Button>
                </DialogContent>
              </Dialog>
            </motion.div>
          )}

          {currentStep === 1 && (
            <motion.div
              key="step1"
              className="w-1/3 mx-auto flex-col justify-center space-y-6 items-start py-24"
              variants={fadeVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              transition={{ duration: 0.2 }}
            >
              <p className="font-circular font-bold text-4xl tracking-tight">What makes you thrive?</p>
              <div className="flex flex-col space-y-4">
                <p>To have a meaningful life and career, it is important to know what matters to you.</p>
                <p>This assessment helps you identify and prioritize what is essential to you at your core, work cultures that fit for you, and skills you enjoy using. This is a practical tool that will give you concrete information you can use to understand and tell your story.</p>
                <p>You will be sorting three categories of cards: Skills, Work Culture, and Core Values.</p>
                <p>Consider what is most important to you in an ideal work situation. Use this frame of reference when you are sorting the cards. You do not need to think about a specific job.</p>
                <p>You get to define what each word or phrase means to you. There is no need to follow the standard definition of a word.</p>
              </div>
              <div className="flex flex-row items-center w-full justify-between">
                <Button variant="secondary" onClick={handleBack}>Back</Button>
                <div className="flex flex-row space-x-2 items-center">
                    {[1, 2, 3, 4, 5, 6].map((step) => (
                        <button
                            key={step}
                            onClick={() => setCurrentStep(step)}
                            className={`w-[36px] h-[36px] flex items-center justify-center rounded-lg
                                ${currentStep === step ? 'bg-tertiaryBlue text-primaryBlue' : 'bg-secondaryBlue text-white'}`}
                        >
                            {step}
                        </button>
                    ))}
                </div>
                <Button variant="secondary" onClick={handleNext}>Next</Button>
              </div>
            </motion.div>
          )}

          {currentStep === 2 && (
            <motion.div
              key="step2"
              className="w-1/3 mx-auto flex-col justify-center space-y-6 items-start py-24"
              variants={fadeVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              transition={{ duration: 0.2 }}
            >
              <p className="font-circular font-bold text-4xl tracking-tight">The basic steps</p>
              <div className="flex flex-col space-y-4">
                <p>1. You will be sorting each card, one category at a time, into High, Medium, and Low areas to indicate the level of importance and/or enjoyment for you. The number of cards you put in each area is up to you.</p>
                <p>2. To move the cards around for each category, you can drag them or you can hover over the card and use the arrow menu.</p>
                <p>3. For each category, you'll take your High cards and narrow them down to your top 5.</p>
                <p>4. Next you'll rank order your top 5 from 1 to 5. If you are having trouble ranking the cards, focus on what energizes you. You will come back to these top 5 lists.</p>
                <p>5. At the end of the assessment you will have the opportunity to download a PDF of your results.</p>
              </div>
              <div className="flex flex-row items-center w-full justify-between">
                <Button variant="secondary" onClick={handleBack}>Back</Button>
                <div className="flex flex-row space-x-2 items-center">
                    {[1, 2, 3, 4, 5, 6].map((step) => (
                        <button
                            key={step}
                            onClick={() => setCurrentStep(step)}
                            className={`w-[36px] h-[36px] flex items-center justify-center rounded-lg
                                ${currentStep === step ? 'bg-tertiaryBlue text-primaryBlue' : 'bg-secondaryBlue text-white'}`}
                        >
                            {step}
                        </button>
                    ))}
                </div>
                <Button variant="secondary" onClick={handleNext}>Next</Button>
              </div>
            </motion.div>
          )}

          {currentStep === 3 && (
            <motion.div
              key="step3"
              className="mx-auto flex-col justify-center space-y-6 items-start py-12 w-[92%]"
              variants={fadeVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              transition={{ duration: 0.2 }}
            >
              <div>
                <p className="font-circular font-bold text-4xl tracking-tight mb-2">Skills</p>
                <p className="font-circular font-medium text-xl tracking-tight "><span className="font-bold text-tertiaryBlue">Guiding Question: </span>What skills would I like to be using most at work?</p>
                <p className="font-circular font-medium text-xl tracking-tight"><span className="font-bold text-tertiaryBlue">Action: </span>Sort the skills cards according to the level of enjoyment for you. Don't worry about your skill level.</p>

              </div>
              <div className="flex flex-row space-x-4">
                <div className="flex flex-col space-y-3 pt-4">
                  <div className="w-[100px] h-[100px] rounded-2xl bg-gray-700 flex items-center justify-center text-white font-bold">
                    High
                  </div>
                  <div className="w-[100px] h-[100px] rounded-2xl bg-gray-700 flex items-center justify-center text-white font-bold">
                    Medium
                  </div>
                  <div className="w-[100px] h-[100px] rounded-2xl bg-gray-700 flex items-center justify-center text-white font-bold">
                    Low
                  </div>
                </div>
                <SkillsGrid 
                    skills={skills} 
                    onHighPrioritySkillsChange={(skills) => setStep3HighSkills(skills)}
                    onTopTenChange={setTopSkills}
                />
              </div>
              <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>Instructions for Skills Sorting</DialogTitle>
                    <DialogDescription>
                      Take your time to sort each skill based on how much you enjoy using it. Don't worry about your current proficiency level - focus on what you'd like to be doing in your ideal role.
                      Drag each card to the appropriate column (High, Medium, or Low) based on your level of enjoyment.
                    </DialogDescription>
                  </DialogHeader>
                  <Button variant="secondary" onClick={() => setDialogOpen(false)}>Got it</Button>
                </DialogContent>
              </Dialog>

              <div className="flex flex-row items-center w-full justify-between mt-8">
                <Button variant="secondary" onClick={handleBack}>Back</Button>
                <div className="flex flex-row space-x-2 items-center">
                    {[1, 2, 3, 4, 5, 6].map((step) => (
                        <button
                            key={step}
                            onClick={() => setCurrentStep(step)}
                            className={`w-[36px] h-[36px] flex items-center justify-center rounded-lg
                                ${currentStep === step ? 'bg-tertiaryBlue text-primaryBlue' : 'bg-secondaryBlue text-white'}`}
                        >
                            {step}
                        </button>
                    ))}
                </div>
                <Button variant="secondary" onClick={handleNext}>Next</Button>
              </div>
            </motion.div>
          )}

          {currentStep === 4 && (
            <motion.div
              key="step4"
              className="mx-auto flex-col justify-center space-y-6 items-start py-12 w-[92%]"
              variants={fadeVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              transition={{ duration: 0.2 }}
            >
                <div>
                    <p className="font-circular font-bold text-4xl tracking-tight mb-2">Work culture</p>
                    <p className="font-circular font-medium text-xl tracking-tight "><span className="font-bold text-tertiaryBlue">Guiding Question: </span>What motivates me and helps me thrive in a work environment?</p>
                    <p className="font-circular font-medium text-xl tracking-tight"><span className="font-bold text-tertiaryBlue">Action: </span>Sort the work culture cards according to the level of importance for you.</p>

                </div>
                <div className="flex flex-row space-x-4">
                    <div className="flex flex-col space-y-3 pt-4">
                    <div className="w-[100px] h-[100px] rounded-2xl bg-gray-700 flex items-center justify-center text-white font-bold">
                        High
                    </div>
                    <div className="w-[100px] h-[100px] rounded-2xl bg-gray-700 flex items-center justify-center text-white font-bold">
                        Medium
                    </div>
                    <div className="w-[100px] h-[100px] rounded-2xl bg-gray-700 flex items-center justify-center text-white font-bold">
                        Low
                    </div>
                    </div>
                    <SkillsGrid skills={workCulture} onTopTenChange={setTopWorkCulture} />
                </div>
                <div className="flex flex-row items-center w-full justify-between mt-8">
                    <Button variant="secondary" onClick={handleBack}>Back</Button>
                    <div className="flex flex-row space-x-2 items-center">
                        {[1, 2, 3, 4, 5, 6].map((step) => (
                            <button
                                key={step}
                                onClick={() => setCurrentStep(step)}
                                className={`w-[36px] h-[36px] flex items-center justify-center rounded-lg
                                    ${currentStep === step ? 'bg-tertiaryBlue text-primaryBlue' : 'bg-secondaryBlue text-white'}`}
                            >
                                {step}
                            </button>
                        ))}
                    </div>
                    <Button variant="secondary" onClick={handleNext}>Next</Button>
                </div>
            </motion.div>
          )}

          {currentStep === 5 && (
            <motion.div
              key="step5"
              className="mx-auto flex-col justify-center space-y-6 items-start py-12 w-[92%]"
              variants={fadeVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              transition={{ duration: 0.2 }}
            >
                <div>
                    <p className="font-circular font-bold text-4xl tracking-tight mb-2">Core values</p>
                    <p className="font-circular font-medium text-xl tracking-tight "><span className="font-bold text-tertiaryBlue">Guiding Question: </span>What is integral to how I define my way of being and live my life?</p>
                    <p className="font-circular font-medium text-xl tracking-tight"><span className="font-bold text-tertiaryBlue">Action: </span>Sort the core values cards according to the level of importance for you. Consider what is essential to you.</p>
                </div>
              <div className="flex flex-row space-x-4">
                <div className="flex flex-col space-y-3 pt-4">
                  <div className="w-[100px] h-[100px] rounded-2xl bg-gray-700 flex items-center justify-center text-white font-bold">
                    High
                  </div>
                  <div className="w-[100px] h-[100px] rounded-2xl bg-gray-700 flex items-center justify-center text-white font-bold">
                    Medium
                  </div>
                  <div className="w-[100px] h-[100px] rounded-2xl bg-gray-700 flex items-center justify-center text-white font-bold">
                    Low
                  </div>
                </div>
                <SkillsGrid skills={coreValues} onTopTenChange={setTopCoreValues} />
              </div>
              <div className="flex flex-row items-center w-full justify-between mt-8">
                <Button variant="secondary" onClick={handleBack}>Back</Button>
                <div className="flex flex-row space-x-2 items-center">
                    {[1, 2, 3, 4, 5, 6].map((step) => (
                        <button
                            key={step}
                            onClick={() => setCurrentStep(step)}
                            className={`w-[36px] h-[36px] flex items-center justify-center rounded-lg
                                ${currentStep === step ? 'bg-tertiaryBlue text-primaryBlue' : 'bg-secondaryBlue text-white'}`}
                        >
                            {step}
                        </button>
                    ))}
                </div>
                <Button variant="secondary" onClick={handleNext}>Next</Button>
              </div>
            </motion.div>
          )}

          {currentStep === 6 && (
            <motion.div
              key="step6"
              className="w-1/3 mx-auto flex-col justify-center space-y-6 items-start py-24"
              variants={fadeVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              transition={{ duration: 0.2 }}
            >
              <p className="font-circular font-bold text-4xl tracking-tight">Anything else?</p>
              <div className="flex flex-col space-y-4 pb-24">
                <Textarea 
                  className="min-h-[250px]" 
                  placeholder="Share any additional thoughts, questions, or context about your career journey..."
                  value={additionalInterests}
                  onChange={(e) => setAdditionalInterests(e.target.value)}
                />
              </div>
              <div className="flex flex-row items-center w-full justify-between">
                <Button variant="secondary" onClick={handleBack}>Back</Button>
                <div className="flex flex-row space-x-2 items-center">
                    {[1, 2, 3, 4, 5, 6].map((step) => (
                        <button
                            key={step}
                            onClick={() => setCurrentStep(step)}
                            className={`w-[36px] h-[36px] flex items-center justify-center rounded-lg
                                ${currentStep === step ? 'bg-tertiaryBlue text-primaryBlue' : 'bg-secondaryBlue text-white'}`}
                        >
                            {step}
                        </button>
                    ))}
                </div>
                <Button variant="secondary" onClick={handleFinish}>Finish</Button>
              </div>
            </motion.div>
          )}

        </AnimatePresence>
      </main>
    </div>
  );
}
