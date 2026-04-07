import { useEffect, useRef } from 'react';
import { Sprout, Database, Target, Bot, BarChart3 } from 'lucide-react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

const steps = [
  {
    num: '01',
    icon: Sprout,
    title: 'Seed Program',
    description: 'Start with a seed program, task prompt, and evaluator that together define the search objective.',
  },
  {
    num: '02',
    icon: Database,
    title: 'Population Database',
    description: 'Keep a ranked database of programs, scores, and metrics so later prompts learn from strong runs.',
  },
  {
    num: '03',
    icon: Target,
    title: 'Sample Inspirations',
    description: 'Sample strong and diverse inspirations using balance-style selection, with extensions for chains.',
  },
  {
    num: '04',
    icon: Bot,
    title: 'LLM Generate',
    description: 'Send the task, inspirations, and metrics to an LLM so it proposes a stronger full program.',
  },
  {
    num: '05',
    icon: BarChart3,
    title: 'Evaluate & Loop',
    description: 'Evaluate candidates, keep the best updates, and checkpoint progress for longer async searches.',
  },
];

export function HowItWorks() {
  const sectionRef = useRef<HTMLDivElement>(null);
  const stepsRef = useRef<HTMLDivElement>(null);
  const lineRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const ctx = gsap.context(() => {
      gsap.fromTo(
        '.hiw-title',
        { y: 40, opacity: 0 },
        {
          y: 0,
          opacity: 1,
          duration: 0.8,
          ease: 'power3.out',
          scrollTrigger: {
            trigger: sectionRef.current,
            start: 'top 80%',
          },
        }
      );

      gsap.fromTo(
        lineRef.current,
        { scaleX: 0 },
        {
          scaleX: 1,
          duration: 1.5,
          ease: 'power2.inOut',
          scrollTrigger: {
            trigger: stepsRef.current,
            start: 'top 80%',
          },
        }
      );

      gsap.fromTo(
        stepsRef.current?.children || [],
        { y: 50, opacity: 0, scale: 0.9 },
        {
          y: 0,
          opacity: 1,
          scale: 1,
          duration: 0.8,
          ease: 'power3.out',
          stagger: 0.15,
          scrollTrigger: {
            trigger: stepsRef.current,
            start: 'top 80%',
          },
        }
      );
    }, sectionRef);

    return () => ctx.revert();
  }, []);

  return (
    <section id="how-it-works" ref={sectionRef} className="relative py-24 bg-black">
      {/* Background */}
      <div className="absolute inset-0 geometric-grid opacity-30" />
      <div className="absolute inset-0 network-lines" />

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="hiw-title text-center mb-20">
          <div className="inline-flex items-center gap-2 mb-4">
            <div className="w-8 h-0.5 bg-cyan-500/50" />
            <span className="text-sm font-medium text-cyan-400 uppercase tracking-wider">Architecture</span>
            <div className="w-8 h-0.5 bg-cyan-500/50" />
          </div>
          <h2 className="text-3xl sm:text-4xl font-bold text-white mb-4">How It Works</h2>
          <p className="text-slate-400 max-w-2xl mx-auto">
            An evolutionary loop that grows stronger programs by sampling inspirations, 
            generating candidates, evaluating them asynchronously, and checkpointing the best discoveries.
          </p>
        </div>

        {/* Steps */}
        <div className="relative pb-8">
          {/* Connecting line - desktop only */}
          <div
            ref={lineRef}
            className="hidden lg:block absolute top-6 left-[10%] right-[10%] h-0.5 
                       bg-gradient-to-r from-cyan-500/20 via-cyan-400/40 to-cyan-500/20 origin-left z-0"
            style={{ transform: 'scaleX(0)' }}
          />

          {/* Steps Grid */}
          <div ref={stepsRef} className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-8 relative z-10">
            {steps.map((step, index) => (
              <div key={index} className="relative group">
                {/* Step number */}
                <div
                  className="w-12 h-12 rounded-full bg-cyan-500 flex items-center justify-center mb-6 mx-auto
                             group-hover:scale-110 group-hover:shadow-lg group-hover:shadow-cyan-500/30
                             transition-all duration-300"
                >
                  <span className="text-black font-mono font-bold text-sm">
                    {step.num}
                  </span>
                </div>

                {/* Card */}
                <div
                  className="p-6 rounded-xl bg-cyan-500/[0.02] border border-cyan-500/10 
                             hover:border-cyan-500/30 hover:bg-cyan-500/[0.04]
                             transition-all duration-300 h-full"
                >
                  {/* Icon */}
                  <div className="mb-4">
                    <step.icon
                      className="w-8 h-8 text-slate-400 group-hover:text-cyan-400 
                                 group-hover:rotate-12 transition-all duration-300"
                    />
                  </div>

                  {/* Content */}
                  <h3 className="text-lg font-semibold text-white mb-2">
                    {step.title}
                  </h3>
                  <p className="text-sm text-slate-500 leading-relaxed">
                    {step.description}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Loop indicator - with proper spacing */}
        <div className="pt-16 text-center">
          <div className="inline-flex items-center gap-3 px-6 py-3 rounded-full bg-cyan-500/5 border border-cyan-500/20">
            <svg
              className="w-5 h-5 text-cyan-400 animate-spin"
              style={{ animationDuration: '3s' }}
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8" />
              <path d="M21 3v5h-5" />
            </svg>
            <span className="text-sm text-slate-400">
              Steps 02 to 05 repeat until the score target is met or budget is exhausted
            </span>
          </div>
        </div>
      </div>
    </section>
  );
}
