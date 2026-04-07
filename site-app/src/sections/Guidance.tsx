import { useEffect, useRef } from 'react';
import { BookOpen, FileText, Github, ArrowRight } from 'lucide-react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

const withBase = (path: string) => `${import.meta.env.BASE_URL}${path.replace(/^\//, '')}`;

const guidanceItems = [
  {
    icon: BookOpen,
    title: 'Documentation',
    description: 'Installation guide, quickstart tutorial, CLI reference, and architecture deep-dive.',
    href: withBase('guides/documentation/'),
  },
  {
    icon: FileText,
    title: 'Technical Report',
    description: 'Read the technical report explaining the method, training setup, and benchmark results.',
    href: '#',
  },
  {
    icon: Github,
    title: 'Source Code',
    description: 'Browse the source code, scripts, datasets, and benchmark tasks on GitHub.',
    href: 'https://github.com/luoqm6will/SimpleEvolve',
  },
];

export function Guidance() {
  const sectionRef = useRef<HTMLDivElement>(null);
  const cardsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const ctx = gsap.context(() => {
      gsap.fromTo(
        '.guidance-title',
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
        cardsRef.current?.children || [],
        { y: 60, opacity: 0, scale: 0.95 },
        {
          y: 0,
          opacity: 1,
          scale: 1,
          duration: 0.8,
          ease: 'power3.out',
          stagger: 0.15,
          scrollTrigger: {
            trigger: cardsRef.current,
            start: 'top 80%',
          },
        }
      );
    }, sectionRef);

    return () => ctx.revert();
  }, []);

  return (
    <section id="guidance" ref={sectionRef} className="relative py-24 bg-[#050a14]">
      {/* Background */}
      <div className="absolute inset-0 geometric-grid opacity-30" />
      <div className="absolute inset-0 network-lines" />

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="guidance-title text-center mb-16">
          <div className="inline-flex items-center gap-2 mb-4">
            <div className="w-8 h-0.5 bg-cyan-500/50" />
            <span className="text-sm font-medium text-cyan-400 uppercase tracking-wider">Resources</span>
            <div className="w-8 h-0.5 bg-cyan-500/50" />
          </div>
          <h2 className="text-3xl sm:text-4xl font-bold text-white mb-4">Jump Right In</h2>
          <p className="text-slate-400 max-w-2xl mx-auto">
            Repo docs, quickstart guides, and report entry points for understanding 
            what SimpleEvolve does and why it matters.
          </p>
        </div>

        {/* Cards Grid */}
        <div ref={cardsRef} className="grid md:grid-cols-3 gap-6">
          {guidanceItems.map((item, index) => (
            <a
              key={index}
              href={item.href}
              target={item.href.startsWith('http') ? '_blank' : undefined}
              rel={item.href.startsWith('http') ? 'noopener noreferrer' : undefined}
              className="group relative"
            >
              <div
                className="relative h-full p-8 rounded-2xl bg-cyan-500/[0.02] border border-cyan-500/10 
                           backdrop-blur-sm transition-all duration-500 card-lift
                           hover:border-cyan-500/30 hover:shadow-2xl"
              >
                {/* Arrow indicator */}
                <div className="absolute top-6 right-6 opacity-0 group-hover:opacity-100 
                               transform translate-x-2 group-hover:translate-x-0 
                               transition-all duration-300">
                  <ArrowRight className="w-5 h-5 text-cyan-400" />
                </div>

                {/* Icon */}
                <div
                  className="w-14 h-14 rounded-xl bg-cyan-500/10 border border-cyan-500/20 
                             flex items-center justify-center mb-6
                             group-hover:scale-110 group-hover:bg-cyan-500/20 transition-all duration-300"
                >
                  <item.icon className="w-7 h-7 text-cyan-400" />
                </div>

                {/* Content */}
                <h3 className="text-xl font-semibold text-white mb-3 group-hover:text-cyan-300 transition-colors">
                  {item.title}
                </h3>
                <p className="text-slate-500 text-sm leading-relaxed">
                  {item.description}
                </p>

                {/* Hover glow */}
                <div
                  className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 
                             transition-opacity duration-500 pointer-events-none"
                  style={{
                    background: 'radial-gradient(circle at 50% 0%, rgba(6, 182, 212, 0.1), transparent 70%)',
                  }}
                />
              </div>
            </a>
          ))}
        </div>
      </div>
    </section>
  );
}
