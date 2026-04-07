import { Github, BookOpen, MessageCircle } from 'lucide-react';

const withBase = (path: string) => `${import.meta.env.BASE_URL}${path.replace(/^\//, '')}`;

const footerLinks = [
  { label: 'Docs', href: withBase('guides/documentation/'), icon: BookOpen },
  { label: 'GitHub', href: 'https://github.com/luoqm6will/SimpleEvolve', icon: Github },
  { label: 'Issues', href: 'https://github.com/luoqm6will/SimpleEvolve/issues', icon: MessageCircle },
];

export function Footer() {
  return (
    <footer className="relative py-12 bg-black border-t border-white/5">
      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6">
          {/* Logo */}
          <a href="#" className="text-slate-300 hover:text-cyan-400 transition-colors">
            SimpleEvolve
          </a>

          {/* Links */}
          <div className="flex items-center gap-6">
            {footerLinks.map((link) => (
              <a
                key={link.label}
                href={link.href}
                target={link.href.startsWith('http') ? '_blank' : undefined}
                rel={link.href.startsWith('http') ? 'noopener noreferrer' : undefined}
                className="flex items-center gap-2 text-slate-500 hover:text-cyan-400 transition-colors"
              >
                <link.icon className="w-4 h-4" />
                <span className="text-sm">{link.label}</span>
              </a>
            ))}
          </div>

          {/* Copyright */}
          <p className="text-sm text-slate-600">
            © 2025 WILL Lab
          </p>
        </div>
      </div>
    </footer>
  );
}
