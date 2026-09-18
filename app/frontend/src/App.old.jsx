import React, { useState, useEffect } from 'react';

export default function RouteResilienceDashboard() {
  // ========================================================
  // 1. APPLICATION DYNAMIC STATE CONTROLS
  // ========================================================
  
  // Pipeline Viewport Layer matches the stages from Person 1 & Person 3
  const [pipelineLayer, setPipelineLayer] = useState('reconstructed'); // Options: 'raw' | 'mask' | 'reconstructed'
  
  // Disaster scenario triggers corresponding to the calculations of Person 4
  const [chaosScenario, setChaosScenario] = useState('none');          // Options: 'none' | 'flood' | 'earthquake'
  
  // Internal console logger feed to trace interactive workspace updates
  const [liveLog, setLiveLog] = useState([]);

  // ========================================================
  // 2. QUANTITATIVE PIPELINE METRICS GENERATOR
  // ========================================================
  
  // Data values simulated from the graph calculation matrix owned by Person 4
  const metrics = {
    efficiency: chaosScenario === 'none' ? '98.4%' : chaosScenario === 'flood' ? '74.1%' : '52.8%',
    criticalRoads: chaosScenario === 'none' ? '14' : chaosScenario === 'flood' ? '8' : '3',
    isolatedNodes: chaosScenario === 'none' ? '0' : chaosScenario === 'flood' ? '12' : '29',
  };

  // ========================================================
  // 3. LOG FEED RECORDING MECHANISM
  // ========================================================
  
  // Records state updates and pushes descriptive text strings into the tracking panel
  useEffect(() => {
    const timestamp = new Date().toLocaleTimeString();
    let newMsg = `[${timestamp}] Viewport rendered layer: ${pipelineLayer.toUpperCase()}`;
    
    if (chaosScenario !== 'none') {
      newMsg = `[${timestamp}] ⚠️ THREAT ALERT: ${chaosScenario.toUpperCase()} disaster simulation active!`;
    }

    // Retain only the latest 6 items to prevent container layout overflow issues
    setLiveLog(prev => [newMsg, ...prev.slice(0, 5)]);
  }, [chaosScenario, pipelineLayer]);

  // ========================================================
  // 4. THE CORE DASHBOARD GRAPHICAL WORKSPACE INTERFACE
  // ========================================================
  return (
    <div className="flex h-screen bg-[#0b0f19] text-slate-100 font-sans overflow-hidden w-full select-none">
      
      {/* --------------------------------------------------------
          COLUMN 1: LEFT SIDEBAR PANEL (USER PARAMETER CONTROLS)
          -------------------------------------------------------- */}
      <aside className="w-80 bg-[#111827] border-r border-slate-800 flex flex-col p-5 space-y-6 shrink-0">
        
        {/* Module Header & Product Context */}
        <div>
          <div className="flex items-center space-x-2.5 mb-2">
            <div className="bg-indigo-600 p-2 rounded-xl text-white font-black text-xs shadow-md shadow-indigo-600/20">
              🛡️
            </div>
            <div>
              <h1 className="font-black text-base tracking-wider bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">
                RouteResilience
              </h1>
              <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">
                Mumbai Grid Command Center
              </p>
            </div>
          </div>
          <p className="text-xs text-slate-400 leading-relaxed">
            AI-powered road network parsing, shadow occlusion recovery, and simulated structural threat profiling.
          </p>
        </div>

        {/* Dynamic Viewport Toggles (Hooks into Person 1 and Person 3 layers) */}
        <div className="space-y-2 pt-4 border-t border-slate-800/60">
          <label className="text-[11px] font-bold uppercase tracking-wider text-indigo-400">Pipeline Viewport Layer</label>
          <div className="flex flex-col space-y-1.5">
            {[
              { id: 'raw', name: '📷 Raw Satellite Imagery' },
              { id: 'mask', name: '👺 AI Segmented Road Mask' },
              { id: 'reconstructed', name: '🕸️ Recovered Road Graph' }
            ].map(layer => (
              <button
                key={layer.id}
                onClick={() => setPipelineLayer(layer.id)}
                className={`w-full text-left px-3 py-2 rounded-lg text-xs font-medium transition-all ${
                  pipelineLayer === layer.id 
                    ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/20' 
                    : 'bg-slate-900/60 text-slate-400 hover:bg-slate-800 hover:text-slate-200'
                }`}
              >
                {layer.name}
              </button>
            ))}
          </div>
        </div>

        {/* Disaster Scenario Dropdown Selector (Hooks into Person 4 threat variables) */}
        <div className="space-y-2 pt-4 border-t border-slate-800/80">
          <label className="text-[11px] font-bold text-slate-400 uppercase tracking-wider flex items-center">
            🔥 Disaster Failure Vector
          </label>
          <select
            value={chaosScenario}
            onChange={(e) => setChaosScenario(e.target.value)}
            className="w-full bg-[#1f2937] border border-slate-700/60 rounded-xl p-3 text-xs font-semibold text-slate-100 outline-none focus:border-indigo-500 cursor-pointer transition-colors"
          >
            <option value="none">🟢 Clear / Stable Conditions</option>
            <option value="flood">🔵 Localized Flash Flooding</option>
            <option value="earthquake">🔴 Seismic Grid Disruption</option>
          </select>
        </div>

        {/* Terminal History Activity Output Box */}
        <div className="flex-1 flex flex-col min-h-0 pt-4 border-t border-slate-800/60">
          <label className="text-[11px] font-bold uppercase tracking-wider text-slate-400 mb-2">Live Activity Feed</label>
          <div className="flex-1 bg-slate-950/80 border border-slate-800 rounded-xl p-3 font-mono text-[11px] text-slate-400 space-y-1.5 overflow-y-auto scrollbar-none">
            {liveLog.length === 0 ? (
              <span className="text-slate-600 italic">Listening for network grid state changes...</span>
            ) : (
              liveLog.map((log, index) => (
                <div key={index} className={log.includes('⚠️') ? 'text-amber-400 font-bold' : 'text-slate-300'}>
                  {log}
                </div>
              ))
            )}
          </div>
        </div>
      </aside>

      {/* --------------------------------------------------------
          COLUMN 2: CENTER WORKSPACE CANVAS (SPATIAL GEO DISPLAY)
          -------------------------------------------------------- */}
      <main className="flex-1 h-full relative bg-[#090d16] flex flex-col">
        
        {/* Active Structural Component Layer Badge */}
        <div className="absolute top-4 left-4 bg-slate-900/90 border border-slate-800 text-slate-300 px-3 py-1.5 rounded-lg text-xs font-semibold backdrop-blur-sm uppercase tracking-wide z-10">
          Layer: <span className="text-indigo-400 font-bold">{pipelineLayer}</span>
        </div>

        {/* Emergency Disaster Warning Banner Overlay */}
        {chaosScenario !== 'none' && (
          <div className="absolute top-4 right-4 bg-red-950/90 border border-red-500/30 text-red-400 px-4 py-1.5 rounded-lg text-xs font-bold flex items-center space-x-2 animate-pulse backdrop-blur-sm z-10">
            <span className="w-2 h-2 rounded-full bg-red-500"></span>
            <span className="uppercase tracking-wider">Active Failure Mode: {chaosScenario}</span>
          </div>
        )}

        {/* Map Blueprint Simulation Boundary Canvas */}
        <div className="flex-1 m-4 border border-dashed border-slate-800 bg-slate-950/20 rounded-2xl flex flex-col items-center justify-center text-center p-6">
          <div className="w-14 h-14 rounded-2xl bg-indigo-950/50 border border-indigo-500/20 flex items-center justify-center mb-4 text-2xl shadow-inner">
            🗺️
          </div>
          <h3 className="text-sm font-semibold text-slate-300 mb-1">Geospatial Grid Canvas Ready</h3>
          <p className="text-xs text-slate-500 max-w-sm leading-relaxed">
            Displaying topological mappings. The user terminal is configured to render coordinated road arrays passed down from the backend engine.
          </p>
        </div>
      </main>

      {/* --------------------------------------------------------
          COLUMN 3: RIGHT SIDEBAR PANEL (GRID STABILITY METRICS)
          -------------------------------------------------------- */}
      <aside className="w-80 bg-[#111827] border-l border-slate-800 p-5 flex flex-col space-y-6 shrink-0">
        
        <div>
          <h2 className="text-xs font-bold uppercase tracking-wider text-indigo-400 mb-4">Grid Health Metrics</h2>
          <div className="space-y-3">
            
            {/* KPI Stat Block 1: Efficiency */}
            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-3 flex items-center justify-between">
              <span className="text-xs font-medium text-slate-400">Network Mobility Index</span>
              <span className={`text-sm font-black transition-all ${chaosScenario === 'none' ? 'text-emerald-400' : 'text-rose-400'}`}>
                {metrics.efficiency}
              </span>
            </div>
            
            {/* KPI Stat Block 2: Critical Backbones */}
            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-3 flex items-center justify-between">
              <span className="text-xs font-medium text-slate-400">Critical Core Backbones</span>
              <span className="text-sm font-black text-amber-400">{metrics.criticalRoads}</span>
            </div>
            
            {/* KPI Stat Block 3: Isolated Node Intersections */}
