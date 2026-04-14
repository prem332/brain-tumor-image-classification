export default function Header() {
  return (
    <header style={{
      background: '#0d1424',
      borderBottom: '1px solid #1e3a5f',
      position: 'sticky',
      top: 0,
      zIndex: 100
    }}>
      <div style={{
        maxWidth: 680,
        margin: '0 auto',
        padding: '16px 20px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{
            width: 42, height: 42, borderRadius: 10,
            background: 'linear-gradient(135deg, #1d4ed8, #0ea5e9)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            flexShrink: 0
          }}>
            <span style={{ color: 'white', fontWeight: 700, fontSize: 16 }}>BT</span>
          </div>
          <div>
            <h1 style={{ fontSize: 20, fontWeight: 700, color: '#f1f5f9', margin: 0 }}>
              Brain Tumor AI
            </h1>
            <p style={{ fontSize: 13, color: '#64748b', margin: 0 }}>
              VGG16 MRI Classification
            </p>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{
            width: 8, height: 8, borderRadius: '50%',
            background: '#22c55e', display: 'inline-block'
          }}></span>
          <span style={{ fontSize: 13, color: '#64748b' }}>Model Online</span>
        </div>
      </div>
    </header>
  )
}
