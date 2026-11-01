/**
 * ALA LUXURY UI INTERACTION ENGINE
 * Powered by Lenis (Smooth Scroll), GSAP (Motion Architecture), and ReactBits (Spotlights & Magnetic Buttons)
 */

(function () {
  'use strict';

  // 1. Initialize Lenis Smooth Scroll
  let lenis = null;
  if (typeof Lenis !== 'undefined') {
    try {
      lenis = new Lenis({
        duration: 1.2,
        easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
        orientation: 'vertical',
        gestureOrientation: 'vertical',
        smoothWheel: true,
        wheelMultiplier: 1.0,
        touchMultiplier: 1.5,
      });

      // Connect Lenis to requestAnimationFrame
      function raf(time) {
        lenis.raf(time);
        requestAnimationFrame(raf);
      }
      requestAnimationFrame(raf);

      // Integrate with GSAP ScrollTrigger if available
      if (typeof gsap !== 'undefined' && typeof ScrollTrigger !== 'undefined') {
        gsap.registerPlugin(ScrollTrigger);
        lenis.on('scroll', ScrollTrigger.update);
        gsap.ticker.add((time) => {
          lenis.raf(time * 1000);
        });
        gsap.ticker.lagSmoothing(0);
      }
    } catch (e) {
      console.warn('Lenis smooth scroll initialized in fallback mode:', e);
    }
  }

  // 2. ReactBits Spotlight Cards (Mouse Tracker for Radial Sheen)
  function initSpotlightCards() {
    const cards = document.querySelectorAll('.spotlight-card, .feat-card, .review-card, .case-card, .news-card, .art-card');
    cards.forEach((card) => {
      card.addEventListener('pointermove', (e) => {
        const rect = card.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        card.style.setProperty('--mouse-x', `${x}px`);
        card.style.setProperty('--mouse-y', `${y}px`);
      });
      card.addEventListener('pointerleave', () => {
        card.style.setProperty('--mouse-x', '-999px');
        card.style.setProperty('--mouse-y', '-999px');
      });
    });
  }

  // 3. ReactBits Magnetic Buttons
  function initMagneticButtons() {
    const magnets = document.querySelectorAll('.magnetic-btn, .btn-primary, .nav-btn');
    magnets.forEach((btn) => {
      btn.addEventListener('mousemove', (e) => {
        const rect = btn.getBoundingClientRect();
        const x = e.clientX - (rect.left + rect.width / 2);
        const y = e.clientY - (rect.top + rect.height / 2);
        const pull = 0.22;
        btn.style.transform = `translate3d(${x * pull}px, ${y * pull}px, 0)`;
      });
      btn.addEventListener('mouseleave', () => {
        btn.style.transform = 'translate3d(0, 0, 0)';
      });
    });
  }

  // 4. GSAP Luxury Page Entrance & ScrollTrigger Staggers
  function initGsapAnimations() {
    if (typeof gsap === 'undefined') return;

    // Check reduced motion preference
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

    // Hero Timeline
    const heroTl = gsap.timeline({ defaults: { ease: 'power3.out' } });

    if (document.querySelector('header')) {
      heroTl.from('header', { y: -20, opacity: 0, duration: 0.8 });
    }

    if (document.querySelector('.promise-banner')) {
      heroTl.from('.promise-banner', { opacity: 0, y: -10, duration: 0.5 }, '-=0.5');
    }

    if (document.querySelector('.hero h1')) {
      heroTl.from('.hero-tag', { opacity: 0, scale: 0.9, duration: 0.6 }, '-=0.3')
            .from('.hero h1', { opacity: 0, y: 30, duration: 0.9 }, '-=0.4')
            .from('.hero p', { opacity: 0, y: 20, duration: 0.8 }, '-=0.6')
            .from('.hero-actions', { opacity: 0, y: 15, duration: 0.7 }, '-=0.5')
            .from('.hero-guarantee', { opacity: 0, duration: 0.6 }, '-=0.4')
            .from('.art-card', { opacity: 0, scale: 0.95, duration: 1.0 }, '-=0.8');
    }

    if (document.querySelector('.chips a')) {
      heroTl.from('.chips a', {
        opacity: 0,
        y: 12,
        duration: 0.4,
        stagger: 0.05,
      }, '-=0.4');
    }

    // ScrollTrigger Card Staggers
    if (typeof ScrollTrigger !== 'undefined') {
      const sections = [
        { selector: '.features-grid .feat-card', trigger: '.features-grid' },
        { selector: '.reviews-grid .review-card', trigger: '.reviews-grid' },
        { selector: '.case-preview-grid .case-card', trigger: '.case-preview-grid' },
        { selector: '.news-grid .news-card', trigger: '.news-grid' }
      ];

      sections.forEach((sec) => {
        const triggerEl = document.querySelector(sec.trigger);
        const items = document.querySelectorAll(sec.selector);
        if (triggerEl && items.length > 0) {
          gsap.from(items, {
            scrollTrigger: {
              trigger: triggerEl,
              start: 'top 85%',
              toggleActions: 'play none none none',
            },
            opacity: 0,
            y: 35,
            duration: 0.8,
            stagger: 0.1,
            ease: 'power3.out',
          });
        }
      });
    }
  }

  // Run initializers on DOMContentLoaded
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      initSpotlightCards();
      initMagneticButtons();
      initGsapAnimations();
    });
  } else {
    initSpotlightCards();
    initMagneticButtons();
    initGsapAnimations();
  }

  // ============================================================================
  // 5. INTERACTIVE CONSTITUTIONAL LEGAL CONSTELLATION BACKGROUND
  // ============================================================================
  function initLegalConstellation() {
    const canvas = document.getElementById('lux-constellation');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let width = 0, height = 0, dpr = window.devicePixelRatio || 1;
    let mouseX = -9999, mouseY = -9999;
    let targetMouseX = -9999, targetMouseY = -9999;

    const LAW_LABELS = [
      'BNS 2023', 'BNSS 2023', 'BSA 2023', 'Art. 21', 'Art. 14', 'Art. 39A',
      'NALSA 1987', 'RTI 2005', 'CPA 2019', 'IT Act §66D', 'NI Act §138',
      'TPA §106', 'Gratuity Act §4', 'Model Tenancy', 'POSH 2013', 'CPC §80',
      'CrPC Trans.', 'e-Daakhil', '1930 Cyber', 'DLSA Aid', 'Limitation Act',
      'Art. 20(3)', 'Art. 22', 'Ind. Disputes', 'Contract §73', 'Gazette of India'
    ];

    let particles = [];
    const PARTICLE_COUNT = 38;

    function resize() {
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      ctx.scale(dpr, dpr);
    }

    function createParticles() {
      particles = [];
      for (let i = 0; i < PARTICLE_COUNT; i++) {
        const isGold = i % 3 === 0;
        particles.push({
          x: Math.random() * width,
          y: Math.random() * height,
          vx: (Math.random() - 0.5) * 0.45,
          vy: (Math.random() - 0.5) * 0.45,
          radius: isGold ? Math.random() * 2.2 + 2.0 : Math.random() * 1.8 + 1.2,
          color: isGold ? '#e0b04e' : '#5fc4ad',
          halo: isGold ? 'rgba(224, 176, 78, 0.18)' : 'rgba(95, 196, 173, 0.14)',
          label: i < LAW_LABELS.length ? LAW_LABELS[i] : null,
          pulse: Math.random() * Math.PI * 2,
          pulseSpeed: 0.02 + Math.random() * 0.02
        });
      }
    }

    window.addEventListener('resize', () => {
      resize();
      createParticles();
    });

    window.addEventListener('pointermove', (e) => {
      targetMouseX = e.clientX;
      targetMouseY = e.clientY;
    });

    resize();
    createParticles();

    function animate(time) {
      ctx.clearRect(0, 0, width, height);

      // Smooth mouse interpolation
      mouseX += (targetMouseX - mouseX) * 0.08;
      mouseY += (targetMouseY - mouseY) * 0.08;

      // Draw subtle ambient radial glow around mouse
      if (mouseX > 0 && mouseX < width && mouseY > 0 && mouseY < height) {
        const mouseGrad = ctx.createRadialGradient(mouseX, mouseY, 0, mouseX, mouseY, 280);
        mouseGrad.addColorStop(0, 'rgba(95, 196, 173, 0.08)');
        mouseGrad.addColorStop(0.5, 'rgba(224, 176, 78, 0.04)');
        mouseGrad.addColorStop(1, 'transparent');
        ctx.fillStyle = mouseGrad;
        ctx.beginPath();
        ctx.arc(mouseX, mouseY, 280, 0, Math.PI * 2);
        ctx.fill();
      }

      // Update & Draw Connections between nearby particles
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          const maxDist = 135;

          if (dist < maxDist) {
            const alpha = (1 - dist / maxDist) * 0.28;
            ctx.strokeStyle = `rgba(160, 185, 210, ${alpha})`;
            ctx.lineWidth = 0.9;
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.stroke();

            // Luminous golden pulse when near mouse
            const midX = (particles[i].x + particles[j].x) / 2;
            const midY = (particles[i].y + particles[j].y) / 2;
            const mouseDist = Math.hypot(midX - mouseX, midY - mouseY);
            if (mouseDist < 120) {
              const mAlpha = (1 - mouseDist / 120) * 0.45;
              ctx.strokeStyle = `rgba(224, 176, 78, ${mAlpha})`;
              ctx.lineWidth = 1.4;
              ctx.beginPath();
              ctx.moveTo(particles[i].x, particles[i].y);
              ctx.lineTo(particles[j].x, particles[j].y);
              ctx.stroke();
            }
          }
        }
      }

      // Update & Draw Particles
      for (let p of particles) {
        p.pulse += p.pulseSpeed;
        p.x += p.vx;
        p.y += p.vy;

        // Gentle boundary bounce
        if (p.x < 10 || p.x > width - 10) p.vx *= -1;
        if (p.y < 10 || p.y > height - 10) p.vy *= -1;

        // Gentle mouse interaction (soft repulsion)
        const dx = p.x - mouseX;
        const dy = p.y - mouseY;
        const mDist = Math.sqrt(dx * dx + dy * dy);
        if (mDist < 140 && mDist > 0) {
          const force = (140 - mDist) / 140 * 0.7;
          p.x += (dx / mDist) * force * 2;
          p.y += (dy / mDist) * force * 2;
        }

        // Draw particle halo
        const currentRadius = p.radius + Math.sin(p.pulse) * 0.6;
        ctx.fillStyle = p.halo;
        ctx.beginPath();
        ctx.arc(p.x, p.y, currentRadius * 3.5, 0, Math.PI * 2);
        ctx.fill();

        // Draw core particle
        ctx.fillStyle = p.color;
        ctx.beginPath();
        ctx.arc(p.x, p.y, currentRadius, 0, Math.PI * 2);
        ctx.fill();

        // Draw Legal Statute Label on key nodes
        if (p.label) {
          ctx.fillStyle = 'rgba(215, 225, 240, 0.75)';
          ctx.font = '600 10.5px -apple-system, system-ui, sans-serif';
          ctx.fillText(p.label, p.x + 8, p.y + 3);
        }
      }

      requestAnimationFrame(animate);
    }

    requestAnimationFrame(animate);
  }


  // Add to init
  document.addEventListener('DOMContentLoaded', initLegalConstellation);
  if (document.readyState !== 'loading') initLegalConstellation();
})();
