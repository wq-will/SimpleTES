import { useEffect, useRef, useState, useCallback } from 'react';
import { ArrowRight, BookOpen, Github, FileText, ChevronLeft, ChevronRight, X, ZoomIn } from 'lucide-react';
import { Button } from '@/components/ui/button';
import gsap from 'gsap';

const withBase = (path: string) => `${import.meta.env.BASE_URL}${path.replace(/^\//, '')}`;

const resourceLinks = [
  { icon: BookOpen, label: 'Docs', href: withBase('guides/documentation/') },
  { icon: FileText, label: 'Paper', href: '#' },
  { icon: Github, label: 'Code', href: 'https://github.com/luoqm6will/SimpleEvolve' },
];

// Carousel images - can be replaced with actual solution images
const carouselImages = [
  {
    src: withBase('hero-visual.png'),
    alt: 'WILL Lab',
    caption: 'WILL Lab',
  },
  {
    src: withBase('hero-3d.png'),
    alt: 'Evolution Visualization',
    caption: 'Evolution Process',
  },
  {
    src: withBase('hero-bg.jpg'),
    alt: 'Solution Construction',
    caption: 'Solution Construction',
  },
];

export function Hero() {
  const heroRef = useRef<HTMLDivElement>(null);
  const titleRef = useRef<HTMLHeadingElement>(null);
  const subtitleRef = useRef<HTMLParagraphElement>(null);
  const carouselRef = useRef<HTMLDivElement>(null);
  const buttonsRef = useRef<HTMLDivElement>(null);
  
  const [currentSlide, setCurrentSlide] = useState(0);
  const [isAutoPlaying, setIsAutoPlaying] = useState(true);
  const [lightboxOpen, setLightboxOpen] = useState(false);
  const autoPlayRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Auto-play carousel
  useEffect(() => {
    const startAutoPlay = () => {
      if (autoPlayRef.current) clearInterval(autoPlayRef.current);
      autoPlayRef.current = setInterval(() => {
        setCurrentSlide((prev) => (prev + 1) % carouselImages.length);
      }, 4000);
    };
    
    const stopAutoPlay = () => {
      if (autoPlayRef.current) {
        clearInterval(autoPlayRef.current);
        autoPlayRef.current = null;
      }
    };
    
    if (isAutoPlaying && !lightboxOpen) {
      startAutoPlay();
    } else {
      stopAutoPlay();
    }
    
    return stopAutoPlay;
  }, [isAutoPlaying, lightboxOpen]);

  // Slide transition animation
  useEffect(() => {
    if (carouselRef.current) {
      const slides = carouselRef.current.querySelectorAll('.carousel-slide');
      slides.forEach((slide, index) => {
        const isActive = index === currentSlide;
        gsap.to(slide, {
          opacity: isActive ? 1 : 0,
          x: isActive ? 0 : (index < currentSlide ? -50 : 50),
          scale: isActive ? 1 : 0.9,
          duration: 0.5,
          ease: 'power2.out',
        });
      });
    }
  }, [currentSlide]);

  const goToSlide = useCallback((index: number) => {
    setCurrentSlide(index);
    setIsAutoPlaying(false);
    setTimeout(() => setIsAutoPlaying(true), 10000);
  }, []);

  const nextSlide = useCallback(() => {
    goToSlide((currentSlide + 1) % carouselImages.length);
  }, [currentSlide, goToSlide]);

  const prevSlide = useCallback(() => {
    goToSlide((currentSlide - 1 + carouselImages.length) % carouselImages.length);
  }, [currentSlide, goToSlide]);

  const openLightbox = () => {
    setLightboxOpen(true);
    setIsAutoPlaying(false);
  };

  const closeLightbox = () => {
    setLightboxOpen(false);
    setIsAutoPlaying(true);
  };

  useEffect(() => {
    const ctx = gsap.context(() => {
      gsap.fromTo(
        titleRef.current,
        { y: 60, opacity: 0 },
        { y: 0, opacity: 1, duration: 1, ease: 'power3.out', delay: 0.2 }
      );

      gsap.fromTo(
        subtitleRef.current,
        { y: 40, opacity: 0, filter: 'blur(10px)' },
        { y: 0, opacity: 1, filter: 'blur(0px)', duration: 0.8, ease: 'power2.out', delay: 0.4 }
      );

      gsap.fromTo(
        carouselRef.current,
        { scale: 0.8, opacity: 0 },
        { scale: 1, opacity: 1, duration: 1.2, ease: 'back.out(1.7)', delay: 0.1 }
      );

      gsap.fromTo(
        buttonsRef.current?.children || [],
        { y: 30, opacity: 0, scale: 0.9 },
        { y: 0, opacity: 1, scale: 1, duration: 0.6, ease: 'back.out(1.7)', stagger: 0.1, delay: 0.6 }
      );
    }, heroRef);

    return () => ctx.revert();
  }, []);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!carouselRef.current || lightboxOpen) return;
      const { clientX, clientY } = e;
      const { innerWidth, innerHeight } = window;
      const x = (clientX / innerWidth - 0.5) * 10;
      const y = (clientY / innerHeight - 0.5) * 10;
      
      gsap.to(carouselRef.current, {
        rotateY: x,
        rotateX: -y,
        duration: 0.5,
        ease: 'power2.out',
      });
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, [lightboxOpen]);

  // Close lightbox on escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && lightboxOpen) {
        closeLightbox();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [lightboxOpen]);

  return (
    <section
      ref={heroRef}
      className="relative min-h-screen flex items-center pt-16 overflow-hidden bg-black"
    >
      {/* Geometric grid background */}
      <div className="absolute inset-0 geometric-grid opacity-50" />
      
      {/* Cyan glow orbs */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl animate-pulse-glow" />
      <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-cyan-400/5 rounded-full blur-3xl animate-pulse-glow" style={{ animationDelay: '2s' }} />

      {/* Network lines overlay */}
      <div className="absolute inset-0 network-lines" />

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="grid lg:grid-cols-2 gap-12 lg:gap-16 items-center">
          {/* Left: Content */}
          <div className="text-center lg:text-left">
            {/* Title */}
            <h1
              ref={titleRef}
              className="text-4xl sm:text-5xl lg:text-6xl font-bold text-white leading-tight mb-6"
            >
              SimpleEvolve: Evolve Code with{' '}
              <span className="gradient-text">Large Language Models</span>
            </h1>

            {/* Subtitle */}
            <p
              ref={subtitleRef}
              className="text-lg text-slate-400 max-w-xl mx-auto lg:mx-0 mb-8 leading-relaxed"
            >
              SimpleEvolve is a lightweight framework for evolving programs with LLMs. 
              Combine inspiration sampling, prompt-based mutation, and evaluator feedback 
              to rapidly test program-improvement workflows.
            </p>

            {/* CTA Buttons */}
            <div ref={buttonsRef} className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start mb-8">
              <Button
                size="lg"
                className="bg-cyan-500 text-black hover:bg-cyan-400 border-0 text-base px-8 font-bold"
                asChild
              >
                <a href="https://github.com/luoqm6will/SimpleEvolve" target="_blank" rel="noopener noreferrer">
                  Get Started
                  <ArrowRight className="w-5 h-5 ml-2" />
                </a>
              </Button>
            </div>

            {/* Resource Links */}
            <div className="flex flex-wrap gap-3 justify-center lg:justify-start">
              {resourceLinks.map((link, index) => (
                <a
                  key={index}
                  href={link.href}
                  target={link.href.startsWith('http') ? '_blank' : undefined}
                  rel={link.href.startsWith('http') ? 'noopener noreferrer' : undefined}
                  className="group flex items-center gap-2 px-5 py-2.5 rounded-xl 
                             bg-gradient-to-b from-white/[0.08] to-white/[0.02]
                             border border-white/10 
                             hover:border-cyan-500/40 hover:from-cyan-500/15 hover:to-cyan-500/5
                             shadow-[0_2px_8px_-2px_rgba(0,0,0,0.5)]
                             hover:shadow-[0_4px_16px_-4px_rgba(6,182,212,0.3)]
                             hover:-translate-y-0.5
                             transition-all duration-300 ease-out"
                >
                  <link.icon className="w-4 h-4 text-slate-400 group-hover:text-cyan-400 transition-colors duration-300" />
                  <span className="text-sm font-medium text-slate-300 group-hover:text-cyan-300 transition-colors duration-300">
                    {link.label}
                  </span>
                </a>
              ))}
            </div>
          </div>

          {/* Right: Image Carousel */}
          <div className="relative flex justify-center lg:justify-end" style={{ perspective: '1000px' }}>
            <div 
              ref={carouselRef}
              className="relative w-full max-w-md lg:max-w-lg"
              style={{ transformStyle: 'preserve-3d' }}
            >
              {/* Glow effect behind carousel */}
              <div className="absolute inset-0 bg-cyan-500/20 rounded-2xl blur-3xl scale-75 animate-pulse-glow" />
              
              {/* Carousel container */}
              <div className="relative aspect-[4/3] rounded-2xl overflow-hidden bg-black/50 border border-white/10 group">
                {carouselImages.map((image, index) => (
                  <div
                    key={index}
                    className={`carousel-slide absolute inset-0 flex flex-col items-center justify-center p-6
                      ${index === 0 ? 'opacity-100' : 'opacity-0'}`}
                  >
                    <img
                      src={image.src}
                      alt={image.alt}
                      className="w-full h-full object-contain animate-float cursor-pointer"
                      onClick={openLightbox}
                    />
                  </div>
                ))}
                
                {/* Zoom indicator on hover */}
                <div 
                  className="absolute inset-0 flex items-center justify-center bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity duration-300 cursor-pointer z-5"
                  onClick={openLightbox}
                >
                  <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-black/60 backdrop-blur-sm border border-white/20">
                    <ZoomIn className="w-5 h-5 text-white" />
                    <span className="text-sm text-white">Click to enlarge</span>
                  </div>
                </div>
                
                {/* Navigation arrows */}
                <button
                  onClick={prevSlide}
                  className="absolute left-3 top-1/2 -translate-y-1/2 w-10 h-10 rounded-full 
                             bg-black/60 backdrop-blur-sm border border-white/20
                             flex items-center justify-center
                             text-white/70 hover:text-white hover:bg-black/80
                             transition-all duration-300 z-10"
                  aria-label="Previous slide"
                >
                  <ChevronLeft className="w-5 h-5" />
                </button>
                <button
                  onClick={nextSlide}
                  className="absolute right-3 top-1/2 -translate-y-1/2 w-10 h-10 rounded-full 
                             bg-black/60 backdrop-blur-sm border border-white/20
                             flex items-center justify-center
                             text-white/70 hover:text-white hover:bg-black/80
                             transition-all duration-300 z-10"
                  aria-label="Next slide"
                >
                  <ChevronRight className="w-5 h-5" />
                </button>
                
                {/* Slide indicators */}
                <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex gap-2 z-10">
                  {carouselImages.map((_, index) => (
                    <button
                      key={index}
                      onClick={() => goToSlide(index)}
                      className={`w-2 h-2 rounded-full transition-all duration-300
                        ${index === currentSlide 
                          ? 'w-6 bg-cyan-400' 
                          : 'bg-white/30 hover:bg-white/50'
                        }`}
                      aria-label={`Go to slide ${index + 1}`}
                    />
                  ))}
                </div>
                
                {/* Caption */}
                <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/80 to-transparent">
                  <p className="text-center text-sm text-slate-300">
                    {carouselImages[currentSlide].caption}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Lightbox Modal */}
      {lightboxOpen && (
        <div 
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/95 backdrop-blur-sm"
          onClick={closeLightbox}
        >
          {/* Close button */}
          <button
            onClick={closeLightbox}
            className="absolute top-6 right-6 w-12 h-12 rounded-full 
                       bg-white/10 backdrop-blur-sm border border-white/20
                       flex items-center justify-center
                       text-white hover:bg-white/20 hover:scale-110
                       transition-all duration-300 z-50"
            aria-label="Close lightbox"
          >
            <X className="w-6 h-6" />
          </button>
          
          {/* Image navigation */}
          <button
            onClick={(e) => { e.stopPropagation(); prevSlide(); }}
            className="absolute left-6 top-1/2 -translate-y-1/2 w-14 h-14 rounded-full 
                       bg-white/10 backdrop-blur-sm border border-white/20
                       flex items-center justify-center
                       text-white hover:bg-white/20 hover:scale-110
                       transition-all duration-300 z-50"
            aria-label="Previous image"
          >
            <ChevronLeft className="w-8 h-8" />
          </button>
          <button
            onClick={(e) => { e.stopPropagation(); nextSlide(); }}
            className="absolute right-6 top-1/2 -translate-y-1/2 w-14 h-14 rounded-full 
                       bg-white/10 backdrop-blur-sm border border-white/20
                       flex items-center justify-center
                       text-white hover:bg-white/20 hover:scale-110
                       transition-all duration-300 z-50"
            aria-label="Next image"
          >
            <ChevronRight className="w-8 h-8" />
          </button>
          
          {/* Enlarged image */}
          <div 
            className="relative max-w-[90vw] max-h-[85vh]"
            onClick={(e) => e.stopPropagation()}
          >
            <img
              src={carouselImages[currentSlide].src}
              alt={carouselImages[currentSlide].alt}
              className="max-w-full max-h-[85vh] object-contain rounded-lg"
            />
            {/* Caption */}
            <div className="absolute -bottom-12 left-0 right-0 text-center">
              <p className="text-lg text-slate-300">
                {carouselImages[currentSlide].caption}
              </p>
              <p className="text-sm text-slate-500 mt-1">
                {currentSlide + 1} / {carouselImages.length}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Bottom fade */}
      <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-black to-transparent" />
    </section>
  );
}
