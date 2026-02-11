import { motion } from "framer-motion";

/**
 * Reusable glassmorphism card wrapper.
 * Wraps children in a frosted-glass container with optional click/hover.
 */
export default function GlassCard({
  children,
  className = "",
  onClick,
  hover = true,
  animate = true,
  ...props
}) {
  const Component = animate ? motion.div : "div";

  const animateProps = animate
    ? {
        initial: { opacity: 0, y: 20 },
        animate: { opacity: 1, y: 0 },
        transition: { duration: 0.4, ease: "easeOut" },
      }
    : {};

  return (
    <Component
      className={`
        rounded-3xl
        bg-white/[0.04]
        backdrop-blur-xl
        border border-white/[0.06]
        shadow-2xl shadow-black/30
        ${hover ? "hover:bg-white/[0.08] transition-colors duration-300" : ""}
        ${onClick ? "cursor-pointer" : ""}
        ${className}
      `}
      onClick={onClick}
      {...animateProps}
      {...props}
    >
      {children}
    </Component>
  );
}
