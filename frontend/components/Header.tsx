export default function Header() {
  return (
    <header style={{ background: '#0d1424', borderBottom: '1px solid #1e3a5f' }}>
      <div className="max-w-3xl mx-auto px-4 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg flex items-center justify-center"
            style={{ background: 'linear-gradient(135deg, #1d4ed8, #0ea5e9)' }}>
            <span style={{ color: 'white', fontWeight: 700, fontSize: 13 }}>BT</span>
          </div>
          <div>
            <h1 style={{ fontSize: 16, fontWeight: 600, color: '#f1f5f9' }}>Brain Tumor AI</h1>
            <p style={{ fontSize: 11, color: '#64748b' }}>VGG16 MRI Classification</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#22c55e', display: 'inline-block' }}></span>
          <span style={{ fontSize: 12, color: '#64748b' }}>Model Online</span>
        </div>
      </div>
    </header>
  )
}
