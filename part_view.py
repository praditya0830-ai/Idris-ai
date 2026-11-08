"""Browser page for GET /part-view: type a sentence, Idris builds the part, see it in 3D, download STEP/STL."""
PART_VIEW_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Idris · Part test</title>
<style>
:root{--bg:#0F141B;--panel:#161D27;--line:#263241;--text:#E6EBF2;--text2:#9AA6B6;--accent:#6E9BFF;--ok:#4ADE80;--err:#F87171}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 system-ui,sans-serif}
header{padding:14px 16px;border-bottom:1px solid var(--line);font-weight:600}
main{display:grid;grid-template-columns:minmax(0,380px) 1fr;gap:0;min-height:calc(100vh - 50px)}
@media(max-width:820px){main{grid-template-columns:1fr}#view{height:55vh}}
aside{padding:16px;border-right:1px solid var(--line)}
textarea{width:100%;min-height:72px;background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:10px;padding:10px;font:inherit}
button{margin-top:10px;width:100%;padding:11px;border:0;border-radius:10px;background:var(--accent);color:#0B1220;font-weight:600;font-size:15px}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}.chips span{border:1px solid var(--line);border-radius:999px;padding:4px 10px;font-size:13px;color:var(--text2);cursor:pointer}
pre{white-space:pre-wrap;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px;font-size:13px;margin-top:14px}
a{color:var(--accent)}#view{position:relative;min-height:300px}canvas{display:block}
.err{color:var(--err)}.ok{color:var(--ok)}
</style></head><body>
<header>Idris · test: design a simple part from a sentence</header>
<main><aside>
<textarea id="q" placeholder="e.g. kubus 50 mm dari baja">kubus 50 mm</textarea>
<button id="go">Design</button>
<div class="chips"><span>kubus 50 mm</span><span>balok 100x60x20 aluminium</span><span>plat 80x50x10 dengan 4 lubang 6 mm di pojok, 8 mm dari tepi</span><span>tabung d40 lubang 20 tinggi 60</span><span>bola 25 mm titanium</span></div>
<pre id="out">Type a sentence and press Design.</pre>
</aside><div id="view"></div></main>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from "three";
import {STLLoader} from "three/addons/loaders/STLLoader.js";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";
const view=document.getElementById("view"),out=document.getElementById("out"),q=document.getElementById("q");
const scene=new THREE.Scene();scene.background=new THREE.Color(0x0F141B);
const cam=new THREE.PerspectiveCamera(40,1,0.1,100000);const ren=new THREE.WebGLRenderer({antialias:true});
view.appendChild(ren.domElement);const ctl=new OrbitControls(cam,ren.domElement);
scene.add(new THREE.HemisphereLight(0xffffff,0x334455,1.1));const dl=new THREE.DirectionalLight(0xffffff,1.2);dl.position.set(1,2,3);scene.add(dl);
let mesh=null;
function size(){const w=view.clientWidth,h=Math.max(view.clientHeight,300);ren.setSize(w,h);cam.aspect=w/h;cam.updateProjectionMatrix()}
addEventListener("resize",size);size();(function loop(){requestAnimationFrame(loop);ctl.update();ren.render(scene,cam)})();
function show(url){new STLLoader().load(url,g=>{if(mesh){scene.remove(mesh)}g.computeVertexNormals();g.computeBoundingBox();
 const c=new THREE.Vector3();g.boundingBox.getCenter(c);g.translate(-c.x,-c.y,-c.z);
 mesh=new THREE.Mesh(g,new THREE.MeshStandardMaterial({color:0xB8C4D6,metalness:.35,roughness:.45}));mesh.rotation.x=-Math.PI/2;scene.add(mesh);
 const s=g.boundingBox.getSize(new THREE.Vector3()).length();cam.position.set(s*.9,s*.7,s*.9);ctl.target.set(0,0,0);cam.near=s/100;cam.far=s*20;cam.updateProjectionMatrix()})}
async function go(){out.textContent="Working…";
 try{const r=await fetch("/design-part?q="+encodeURIComponent(q.value));const d=await r.json();
  if(!d.success){out.innerHTML='<span class="err">'+(d.answer||d.error||"failed")+'</span>';return}
  out.textContent=d.answer;out.innerHTML=out.innerHTML.replace(/(https?:\/\/\S+)/g,'<a href="$1">$1</a>');show(d.files.stl_url)}
 catch(e){out.innerHTML='<span class="err">'+e+'</span>'}}
document.getElementById("go").onclick=go;document.querySelectorAll(".chips span").forEach(s=>s.onclick=()=>{q.value=s.textContent;go()});
</script></body></html>"""
