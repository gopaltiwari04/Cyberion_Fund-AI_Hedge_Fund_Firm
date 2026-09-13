"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { useSession, signOut } from "next-auth/react";

const navigation = [
  { href: "/", label: "Overview", icon: "O" },
  { href: "/portfolio", label: "Portfolio", icon: "P" },
  { href: "/markets", label: "Markets", icon: "M" },
  { href: "/research", label: "Research", icon: "R" },
  { href: "/system", label: "System", icon: "S" },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { data: session } = useSession();
  const [profileOpen, setProfileOpen] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const today = new Date().toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });

  useEffect(() => {
    const closeMenu = () => setMenuOpen(false);
    window.addEventListener("popstate", closeMenu);
    return () => window.removeEventListener("popstate", closeMenu);
  }, []);

  if (pathname === "/login") {
    return <>{children}</>;
  }

  return <div className="app-shell">
    {menuOpen ? <button className="menu-backdrop" aria-label="Close navigation" onClick={() => setMenuOpen(false)} /> : null}
    <aside id="main-navigation" className={`sidebar ${menuOpen ? "sidebar-open" : ""}`} aria-label="Main navigation">
      <div className="sidebar-top"><Link className="brand" href="/" onClick={() => setMenuOpen(false)}><span className="brand-mark">AI</span><span className="brand-name">AI Hedge Fund</span></Link><button className="close-menu" aria-label="Close navigation" onClick={() => setMenuOpen(false)}>×</button></div>
      <span className="nav-label">Workspace</span>
      <nav className="nav-list">{navigation.map((item) => <Link key={item.href} className={`nav-item ${pathname === item.href ? "active" : ""}`} href={item.href} aria-current={pathname === item.href ? "page" : undefined} onClick={() => setMenuOpen(false)}><span className="nav-icon">{item.icon}</span><span>{item.label}</span></Link>)}</nav>
      <div className="sidebar-footer"><span className="live-dot" />Production environment<br />Read-only terminal</div>
    </aside>
    <main className="main-content">
      <header className="topbar"><button className="menu-toggle" aria-label="Open navigation" aria-controls="main-navigation" aria-expanded={menuOpen} onClick={() => setMenuOpen(true)}>☰</button><div className="breadcrumb">Workspace <span>/</span> <strong>{navigation.find((item) => item.href === pathname)?.label ?? "Overview"}</strong></div><div className="topbar-right">
  <span className="date-stamp">{today}</span>

  <div className="profile-menu">
    <button
      className="profile-trigger"
      onClick={() => setProfileOpen((open) => !open)}
      aria-expanded={profileOpen}
      aria-label="Open profile menu"
    >
      <div className="profile-trigger-avatar">
        {session?.user?.image ? (
          <img
            src={session.user.image}
            alt={session.user.name ?? "Profile"}
          />
        ) : (
          <span>
            {(session?.user?.name ?? "U").charAt(0).toUpperCase()}
          </span>
        )}
      </div>

      <div className="profile-trigger-info">
        <strong>{session?.user?.name ?? "Cyberion User"}</strong>
        <span>Authenticated</span>
      </div>

      <span className="profile-chevron">
        {profileOpen ? "▲" : "▼"}
      </span>
    </button>

    {profileOpen ? (
      <div className="profile-dropdown">
        <div className="profile-dropdown-header">
          <span>CYBERION IDENTITY</span>
          <strong>{session?.user?.email ?? "Authenticated user"}</strong>
        </div>

        <Link
          href="/profile"
          className="profile-dropdown-item"
          onClick={() => setProfileOpen(false)}
        >
          <span>◉</span>
          <div>
            <strong>Profile</strong>
            <small>Account & identity</small>
          </div>
        </Link>

        <div className="profile-dropdown-divider" />

        <button
          className="profile-dropdown-signout"
          onClick={() => signOut({ callbackUrl: "/login" })}
        >
          <span>↪</span>
          <div>
            <strong>Sign out</strong>
            <small>End current session</small>
          </div>
        </button>
      </div>
    ) : null}
  </div>
</div></header>
      {children}
    </main>
  </div>;
}