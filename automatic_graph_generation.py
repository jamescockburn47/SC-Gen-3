#!/usr/bin/env python3
"""
Automatic Knowledge Graph Generation for Legal Documents
=======================================================

This module automatically generates visual knowledge graphs when documents are 
uploaded and processed, providing immediate visual insights into entity relationships.
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import json
import logging
import io
import base64
import streamlit as st
from typing import Dict, List, Tuple, Optional
import re
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Processing time estimates for different methods
PROCESSING_TIMES = {
    'standard_rag': 0.5,
    'hierarchical_retrieval': 0.1,
    'adaptive_chunking': 0.2, 
    'knowledge_graph': 0.3,
    'late_interaction': 2.5,
    'auto_graph_generation': 1.0
}

print("🚀 Automatic Knowledge Graph Generation Module Loaded")
print(f"⏱️ Processing Time Estimates: {PROCESSING_TIMES}")

class AutoGraphGenerator:
    """Automatically generates knowledge graphs during document processing"""
    
    def __init__(self, matter_id: str):
        self.matter_id = matter_id
        self.rag_path = Path(f"rag_storage/{matter_id}")
        self.graph_path = self.rag_path / "knowledge_graphs"
        self.graph_path.mkdir(exist_ok=True)
        
        # Initialize graph
        self.knowledge_graph = nx.DiGraph()
        self.entity_colors = {
            'claimant': '#FF6B6B',      # Red
            'defendant': '#4ECDC4',     # Teal
            'case_reference': '#45B7D1', # Blue
            'legal_claim': '#96CEB4',   # Green
            'evidence': '#FFEAA7',      # Yellow
            'document': '#DDA0DD',      # Plum
            'legal_document': '#98D8C8', # Mint
            'party': '#F7DC6F',         # Light Yellow
            'reference': '#AED6F1'      # Light Blue
        }
    
    def auto_generate_on_upload(self, documents: List[Dict]) -> Dict:
        """Automatically generate knowledge graph when documents are uploaded"""
        
        logger.info(f"🏗️ Auto-generating knowledge graph for {len(documents)} documents...")
        
        # Clear existing graph
        self.knowledge_graph.clear()
        
        entities_found = []
        relationships_created = []
        processing_stats = {
            'documents_processed': 0,
            'entities_extracted': 0,
            'relationships_created': 0,
            'processing_time': 0
        }
        
        import time
        start_time = time.time()
        
        # Process each document
        for doc in documents:
            try:
                doc_entities = self._extract_entities_from_document(doc)
                entities_found.extend(doc_entities)
                
                # Add document node
                doc_name = doc.get('filename', 'Unknown Document')
                self.knowledge_graph.add_node(
                    doc_name,
                    node_type='document',
                    size=300,
                    doc_id=doc.get('id', '')
                )
                
                # Add entity nodes and relationships
                for entity in doc_entities:
                    entity_name = entity['name']
                    entity_type = entity['type']
                    
                    # Add entity node
                    self.knowledge_graph.add_node(
                        entity_name,
                        node_type=entity_type,
                        confidence=entity['confidence'],
                        size=500
                    )
                    
                    # Add document-entity relationship
                    self.knowledge_graph.add_edge(
                        doc_name,
                        entity_name,
                        relation='contains',
                        confidence=entity['confidence'],
                        weight=entity['confidence']
                    )
                    
                    relationships_created.append({
                        'source': doc_name,
                        'target': entity_name,
                        'relation': 'contains'
                    })
                
                processing_stats['documents_processed'] += 1
                
            except Exception as e:
                logger.error(f"Error processing document {doc.get('filename', 'Unknown')}: {e}")
        
        # Create inter-entity relationships
        inter_relationships = self._create_inter_entity_relationships(entities_found)
        relationships_created.extend(inter_relationships)
        
        # Update statistics
        processing_stats['entities_extracted'] = len(set(e['name'] for e in entities_found))
        processing_stats['relationships_created'] = len(relationships_created)
        processing_stats['processing_time'] = time.time() - start_time
        
        # Generate visual graph
        graph_image = self._generate_visual_graph()
        
        # Save graph data
        self._save_graph_data(entities_found, relationships_created, processing_stats)
        
        logger.info(f"✅ Knowledge graph generated: {processing_stats['entities_extracted']} entities, {processing_stats['relationships_created']} relationships")
        
        return {
            'graph_image': graph_image,
            'entities': entities_found,
            'relationships': relationships_created,
            'stats': processing_stats,
            'graph_object': self.knowledge_graph
        }
    
    def _extract_entities_from_document(self, document: Dict) -> List[Dict]:
        """Extract legal entities from document metadata and filename"""
        
        entities = []
        filename = document.get('filename', '').lower()
        doc_content = document.get('content', '')
        
        # Extract based on filename patterns
        
        # 1. Personal Names (Claimants/Defendants)
        name_patterns = [
            r'elyas\s+abaris',
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)',  # General name pattern
        ]
        
        for pattern in name_patterns:
            matches = re.finditer(pattern, filename, re.IGNORECASE)
            for match in matches:
                name = match.group().title()
                entity_type = 'claimant' if any(word in filename for word in ['claimant', 'plaintiff']) else 'party'
                entities.append({
                    'name': name,
                    'type': entity_type,
                    'confidence': 0.9,
                    'source': 'filename'
                })
        
        # 2. Case References
        case_patterns = [
            r'KB-\d{4}-\d{6}',
            r'[A-Z]{2,4}-\d{4}-\d{6}',
            r'Case\s+No[.:]\s*([A-Z0-9\-/]+)'
        ]
        
        for pattern in case_patterns:
            matches = re.finditer(pattern, filename, re.IGNORECASE)
            for match in matches:
                case_ref = match.group().upper()
                entities.append({
                    'name': case_ref,
                    'type': 'case_reference',
                    'confidence': 0.95,
                    'source': 'filename'
                })
        
        # 3. Document Types
        doc_type_mapping = {
            'claim': ('Particulars of Claim', 'legal_document', 0.9),
            'particulars': ('Particulars of Claim', 'legal_document', 0.9),
            'defence': ('Defence', 'legal_document', 0.9),
            'witness': ('Witness Statement', 'evidence', 0.9),
            'statement': ('Statement', 'evidence', 0.8),
            'order': ('Court Order', 'legal_document', 0.9),
            'judgment': ('Judgment', 'legal_document', 0.95),
            'contract': ('Contract', 'legal_document', 0.85)
        }
        
        for keyword, (entity_name, entity_type, confidence) in doc_type_mapping.items():
            if keyword in filename:
                entities.append({
                    'name': entity_name,
                    'type': entity_type,
                    'confidence': confidence,
                    'source': 'filename'
                })
        
        # 4. Legal Claims (from content if available)
        claim_patterns = [
            r'GDPR\s+breach',
            r'contract\s+breach',
            r'negligence',
            r'defamation',
            r'discrimination',
            r'unfair\s+dismissal'
        ]
        
        combined_text = f"{filename} {doc_content}".lower()
        for pattern in claim_patterns:
            matches = re.finditer(pattern, combined_text, re.IGNORECASE)
            for match in matches:
                claim = match.group().title()
                entities.append({
                    'name': f"{claim} Claim",
                    'type': 'legal_claim',
                    'confidence': 0.8,
                    'source': 'content'
                })
        
        # 5. Organizations
        org_patterns = [
            r'UCL',
            r'University\s+College\s+London',
            r'([A-Z]{2,}\s+(?:Ltd|Limited|plc|PLC|LLP))',
            r'([A-Z][a-z]+\s+(?:University|College|School))'
        ]
        
        for pattern in org_patterns:
            matches = re.finditer(pattern, combined_text, re.IGNORECASE)
            for match in matches:
                org_name = match.group()
                entity_type = 'defendant' if any(word in filename for word in ['defence', 'defendant']) else 'party'
                entities.append({
                    'name': org_name,
                    'type': entity_type,
                    'confidence': 0.85,
                    'source': 'content'
                })
        
        return entities
    
    def _create_inter_entity_relationships(self, entities: List[Dict]) -> List[Dict]:
        """Create relationships between entities based on legal context"""
        
        relationships = []
        entity_dict = {e['name']: e for e in entities}
        
        # Find claimants and defendants
        claimants = [e for e in entities if e['type'] == 'claimant']
        defendants = [e for e in entities if e['type'] == 'defendant']
        case_refs = [e for e in entities if e['type'] == 'case_reference']
        legal_claims = [e for e in entities if e['type'] == 'legal_claim']
        evidence = [e for e in entities if e['type'] == 'evidence']
        
        # Create claimant-defendant relationships
        for claimant in claimants:
            for defendant in defendants:
                self.knowledge_graph.add_edge(
                    claimant['name'],
                    defendant['name'],
                    relation='claims_against',
                    confidence=0.9,
                    weight=0.9
                )
                relationships.append({
                    'source': claimant['name'],
                    'target': defendant['name'],
                    'relation': 'claims_against'
                })
        
        # Create claimant-case relationships
        for claimant in claimants:
            for case_ref in case_refs:
                self.knowledge_graph.add_edge(
                    claimant['name'],
                    case_ref['name'],
                    relation='is_claimant_in',
                    confidence=1.0,
                    weight=1.0
                )
                relationships.append({
                    'source': claimant['name'],
                    'target': case_ref['name'],
                    'relation': 'is_claimant_in'
                })
        
        # Create defendant-case relationships
        for defendant in defendants:
            for case_ref in case_refs:
                self.knowledge_graph.add_edge(
                    defendant['name'],
                    case_ref['name'],
                    relation='is_defendant_in',
                    confidence=1.0,
                    weight=1.0
                )
                relationships.append({
                    'source': defendant['name'],
                    'target': case_ref['name'],
                    'relation': 'is_defendant_in'
                })
        
        # Create claimant-claim relationships
        for claimant in claimants:
            for claim in legal_claims:
                self.knowledge_graph.add_edge(
                    claimant['name'],
                    claim['name'],
                    relation='alleges',
                    confidence=0.8,
                    weight=0.8
                )
                relationships.append({
                    'source': claimant['name'],
                    'target': claim['name'],
                    'relation': 'alleges'
                })
        
        # Create evidence-claim relationships
        for evidence_item in evidence:
            for claim in legal_claims:
                self.knowledge_graph.add_edge(
                    evidence_item['name'],
                    claim['name'],
                    relation='supports',
                    confidence=0.7,
                    weight=0.7
                )
                relationships.append({
                    'source': evidence_item['name'],
                    'target': claim['name'],
                    'relation': 'supports'
                })
        
        return relationships
    
    def _generate_visual_graph(self) -> str:
        """Generate visual knowledge graph as base64 encoded image"""
        
        try:
            plt.figure(figsize=(14, 10))
            
            # Use spring layout for better visualization
            pos = nx.spring_layout(self.knowledge_graph, k=3, iterations=50)
            
            # Draw nodes by type with different colors and sizes
            for node_type, color in self.entity_colors.items():
                nodes = [n for n, d in self.knowledge_graph.nodes(data=True) if d.get('node_type') == node_type]
                if nodes:
                    sizes = [self.knowledge_graph.nodes[n].get('size', 400) for n in nodes]
                    nx.draw_networkx_nodes(
                        self.knowledge_graph, pos,
                        nodelist=nodes,
                        node_color=color,
                        node_size=sizes,
                        alpha=0.8
                    )
            
            # Draw edges with different styles for different relations
            edge_styles = {
                'claims_against': {'style': '-', 'width': 3, 'color': 'red'},
                'is_claimant_in': {'style': '-', 'width': 2, 'color': 'blue'},
                'is_defendant_in': {'style': '-', 'width': 2, 'color': 'teal'},
                'alleges': {'style': '--', 'width': 2, 'color': 'green'},
                'supports': {'style': ':', 'width': 2, 'color': 'orange'},
                'contains': {'style': '-', 'width': 1, 'color': 'gray'}
            }
            
            for relation, style in edge_styles.items():
                edges = [(u, v) for u, v, d in self.knowledge_graph.edges(data=True) if d.get('relation') == relation]
                if edges:
                    nx.draw_networkx_edges(
                        self.knowledge_graph, pos,
                        edgelist=edges,
                        style=style['style'],
                        width=style['width'],
                        edge_color=style['color'],
                        alpha=0.7
                    )
            
            # Draw labels
            nx.draw_networkx_labels(
                self.knowledge_graph, pos,
                font_size=8,
                font_weight='bold'
            )
            
            # Create legend
            legend_elements = []
            for entity_type, color in self.entity_colors.items():
                if any(d.get('node_type') == entity_type for n, d in self.knowledge_graph.nodes(data=True)):
                    legend_elements.append(
                        mpatches.Patch(color=color, label=entity_type.replace('_', ' ').title())
                    )
            
            plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
            
            plt.title(f"Knowledge Graph: {self.matter_id}\n{self.knowledge_graph.number_of_nodes()} entities, {self.knowledge_graph.number_of_edges()} relationships", 
                     fontsize=14, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            
            # Convert to base64 for web display
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Error generating visual graph: {e}")
            return ""
    
    def _save_graph_data(self, entities: List[Dict], relationships: List[Dict], stats: Dict):
        """Save graph data for future use"""
        
        graph_data = {
            'entities': entities,
            'relationships': relationships,
            'stats': stats,
            'graph_structure': {
                'nodes': list(self.knowledge_graph.nodes(data=True)),
                'edges': list(self.knowledge_graph.edges(data=True))
            },
            'generation_timestamp': datetime.now().isoformat()
        }
        
        # Save to JSON file
        graph_file = self.graph_path / "knowledge_graph.json"
        with open(graph_file, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        logger.info(f"📊 Graph data saved to {graph_file}")
    
    def display_in_streamlit(self, graph_result: Dict):
        """Display the knowledge graph in Streamlit interface"""
        
        st.markdown("### 🌐 **Automatic Knowledge Graph Generated**")
        
        # Display statistics
        stats = graph_result['stats']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📄 Documents", stats['documents_processed'])
        with col2:
            st.metric("🔗 Entities", stats['entities_extracted'])
        with col3:
            st.metric("↔️ Relationships", stats['relationships_created'])
        with col4:
            st.metric("⚡ Time", f"{stats['processing_time']:.2f}s")
        
        # Display graph image
        if graph_result['graph_image']:
            st.markdown("#### 📊 **Visual Knowledge Graph**")
            st.image(f"data:image/png;base64,{graph_result['graph_image']}", 
                    caption=f"Knowledge Graph for {self.matter_id}")
        
        # Display entities and relationships in expandable sections
        with st.expander("🔍 **Entities Detected**", expanded=False):
            entities_by_type = {}
            for entity in graph_result['entities']:
                entity_type = entity['type']
                if entity_type not in entities_by_type:
                    entities_by_type[entity_type] = []
                entities_by_type[entity_type].append(entity)
            
            for entity_type, entities in entities_by_type.items():
                st.markdown(f"**{entity_type.replace('_', ' ').title()}:**")
                for entity in entities:
                    confidence = entity['confidence']
                    st.write(f"  • {entity['name']} (confidence: {confidence:.1%})")
        
        with st.expander("🌐 **Relationships Mapped**", expanded=False):
            for rel in graph_result['relationships']:
                st.write(f"🔗 **{rel['source']}** --[{rel['relation']}]--> **{rel['target']}**")

def get_processing_time_estimates() -> Dict[str, float]:
    """Get processing time estimates for different enhancement methods"""
    
    return {
        'standard_rag': 0.5,  # Base RAG processing
        'hierarchical_retrieval': 0.1,  # Document structure analysis
        'adaptive_chunking': 0.2,  # Query type optimization
        'knowledge_graph': 0.3,  # Graph building and traversal
        'late_interaction': 2.5,  # ColBERT model processing
        'auto_graph_generation': 1.0,  # Visual graph creation
        'total_with_all_methods': 4.6  # All methods combined
    }

# Auto-integration with document upload process
def integrate_with_document_upload():
    """Instructions for integrating automatic graph generation with document upload"""
    
    integration_code = '''
# Add to your document upload handler:

def process_uploaded_documents(uploaded_files, matter_id):
    """Enhanced document processing with automatic knowledge graph generation"""
    
    from automatic_graph_generation import AutoGraphGenerator
    
    # Process documents normally
    processed_docs = []
    for file in uploaded_files:
        # Your existing document processing logic
        doc_data = process_single_document(file)
        processed_docs.append(doc_data)
    
    # Auto-generate knowledge graph
    graph_generator = AutoGraphGenerator(matter_id)
    graph_result = graph_generator.auto_generate_on_upload(processed_docs)
    
    # Display graph in Streamlit
    graph_generator.display_in_streamlit(graph_result)
    
    # Store graph for future use
    st.session_state[f'knowledge_graph_{matter_id}'] = graph_result
    
    return processed_docs, graph_result
'''
    
    return integration_code

if __name__ == "__main__":
    # Test automatic graph generation
    print("🧪 Testing Automatic Knowledge Graph Generation...")
    
    # Get processing time estimates
    estimates = get_processing_time_estimates()
    print("⏱️ Processing Time Estimates:")
    for method, time_est in estimates.items():
        print(f"   • {method.replace('_', ' ').title()}: {time_est}s")
    
    print(f"\n🚀 Total Enhanced Processing Time: {estimates['total_with_all_methods']}s")
    print("💡 Recommended: Enable hierarchical + adaptive + knowledge graph by default (0.6s total)")
    
    # Show integration example
    integration = integrate_with_document_upload()
    print(f"\n📋 Integration Code:\n{integration}") 