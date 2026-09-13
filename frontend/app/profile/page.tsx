import { auth, signOut } from "@/auth";

export default async function ProfilePage() {
  const session = await auth();
  const user = session?.user;

  return (
    <main className="profile-page">
      <div className="profile-header">
        <div>
          <span className="profile-eyebrow">ACCOUNT / IDENTITY</span>
          <h1>Profile</h1>
          <p>
            Manage your Cyberion identity and view your investment
            intelligence configuration.
          </p>
        </div>

        <div className="profile-status">
          <span className="profile-status-dot" />
          ACCOUNT ACTIVE
        </div>
      </div>

      <div className="profile-grid">
        {/* Identity */}
        <section className="profile-card identity-card">
          <div className="profile-card-header">
            <span>01</span>
            <span>IDENTITY</span>
          </div>

          <div className="profile-identity">
            <div className="profile-avatar">
              {user?.image ? (
                <img
                  src={user.image}
                  alt={user.name ?? "Profile"}
                />
              ) : (
                <span>
                  {(user?.name ?? "U").charAt(0).toUpperCase()}
                </span>
              )}
            </div>

            <div>
              <h2>{user?.name ?? "Cyberion User"}</h2>
              <p>{user?.email ?? "No email available"}</p>

              <div className="profile-auth-badge">
                <span>✓</span>
                GOOGLE AUTHENTICATED
              </div>
            </div>
          </div>

          <div className="profile-divider" />

          <div className="identity-details">
            <div>
              <span>ACCOUNT TYPE</span>
              <strong>INDIVIDUAL</strong>
            </div>

            <div>
              <span>AUTH PROVIDER</span>
              <strong>GOOGLE</strong>
            </div>

            <div>
              <span>SESSION</span>
              <strong>ACTIVE</strong>
            </div>
          </div>
        </section>

        {/* Account */}
        <section className="profile-card">
          <div className="profile-card-header">
            <span>02</span>
            <span>ACCOUNT</span>
          </div>

          <div className="profile-list">
            <div>
              <span>PROFILE</span>
              <strong>Verified</strong>
            </div>

            <div>
              <span>AUTHENTICATION</span>
              <strong>Google</strong>
            </div>

            <div>
              <span>SECURITY</span>
              <strong className="green-text">Protected</strong>
            </div>

            <div>
              <span>SESSION STATUS</span>
              <strong className="green-text">Active</strong>
            </div>
          </div>
        </section>

        {/* Investment profile */}
        <section className="profile-card">
          <div className="profile-card-header">
            <span>03</span>
            <span>INVESTMENT PROFILE</span>
          </div>

          <div className="profile-list">
            <div>
              <span>STRATEGY</span>
              <strong>AI Quantitative</strong>
            </div>

            <div>
              <span>PREDICTION MODEL</span>
              <strong>XGBoost</strong>
            </div>

            <div>
              <span>PORTFOLIO ENGINE</span>
              <strong>Mean-Variance</strong>
            </div>

            <div>
              <span>MARKET UNIVERSE</span>
              <strong>AAPL · AMZN · GOOGL · MSFT · SPY</strong>
            </div>
          </div>
        </section>

        {/* System */}
        <section className="profile-card system-card">
          <div className="profile-card-header">
            <span>04</span>
            <span>SYSTEM</span>
          </div>

          <div className="system-grid">
            <div className="system-item">
              <span className="system-indicator" />
              <div>
                <span>AI ENGINE</span>
                <strong>ONLINE</strong>
              </div>
            </div>

            <div className="system-item">
              <span className="system-indicator" />
              <div>
                <span>MARKET DATA</span>
                <strong>AVAILABLE</strong>
              </div>
            </div>

            <div className="system-item">
              <span className="system-indicator" />
              <div>
                <span>MODEL</span>
                <strong>READY</strong>
              </div>
            </div>

            <div className="system-item">
              <span className="system-indicator" />
              <div>
                <span>API</span>
                <strong>CONNECTED</strong>
              </div>
            </div>
          </div>
        </section>
      </div>

      {/* Sign out */}
      <section className="profile-security">
        <div>
          <span className="profile-eyebrow">SESSION MANAGEMENT</span>
          <h3>Sign out of Cyberion</h3>
          <p>
            End your current authenticated session on this device.
          </p>
        </div>

        <form
          action={async () => {
            "use server";
            await signOut({ redirectTo: "/login" });
          }}
        >
          <button type="submit" className="profile-signout">
            SIGN OUT
            <span>→</span>
          </button>
        </form>
      </section>
    </main>
  );
}