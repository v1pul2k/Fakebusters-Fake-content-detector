from flask import Flask, request, jsonify
from flask_cors import CORS
import re
from datetime import datetime
import nltk
from collections import Counter

app = Flask(__name__)
CORS(app)

# Suspicious keywords and patterns
CLICKBAIT_WORDS = ['shocking', 'unbelievable', 'you won\'t believe', 'miracle', 
                   'secret', 'they don\'t want you to know', 'breaking']
EMOTIONAL_WORDS = ['outraged', 'furious', 'devastating', 'apocalyptic', 'crisis']
FAKE_INDICATORS = ['unnamed sources', 'sources say', 'according to reports', 
                   'allegedly', 'rumor has it']

def analyze_title(title):
    """Analyze title for clickbait and sensationalism"""
    score = 100
    issues = []
    
    title_lower = title.lower()
    
    # Check for all caps
    if title.isupper() and len(title) > 10:
        score -= 15
        issues.append("Excessive capitalization detected")
    
    # Check for clickbait words
    clickbait_count = sum(1 for word in CLICKBAIT_WORDS if word in title_lower)
    if clickbait_count > 0:
        score -= clickbait_count * 10
        issues.append(f"Contains {clickbait_count} clickbait phrase(s)")
    
    # Check for excessive punctuation
    exclamation_count = title.count('!')
    if exclamation_count > 1:
        score -= exclamation_count * 5
        issues.append("Excessive exclamation marks")
    
    return max(0, score), issues

def analyze_content(content):
    """Analyze article content for credibility indicators"""
    score = 100
    issues = []
    
    content_lower = content.lower()
    
    # Check length
    word_count = len(content.split())
    if word_count < 100:
        score -= 20
        issues.append("Article too short (less than 100 words)")
    
    # Check for emotional manipulation
    emotional_count = sum(1 for word in EMOTIONAL_WORDS if word in content_lower)
    if emotional_count > 3:
        score -= emotional_count * 5
        issues.append(f"High emotional language usage ({emotional_count} instances)")
    
    # Check for vague sourcing
    vague_sources = sum(1 for phrase in FAKE_INDICATORS if phrase in content_lower)
    if vague_sources > 2:
        score -= vague_sources * 10
        issues.append(f"Vague or unnamed sources mentioned ({vague_sources} times)")
    
    # Check for proper citations (URLs or quotes)
    url_pattern = r'https?://[^\s]+'
    quote_pattern = r'"[^"]+"'
    urls = len(re.findall(url_pattern, content))
    quotes = len(re.findall(quote_pattern, content))
    
    if urls == 0 and quotes == 0 and word_count > 200:
        score -= 15
        issues.append("No citations or direct quotes found")
    
    # Grammar and spelling check (basic)
    sentences = content.split('.')
    avg_sentence_length = word_count / max(len(sentences), 1)
    
    if avg_sentence_length < 5:
        score -= 10
        issues.append("Unusually short sentences detected")
    
    return max(0, score), issues

def analyze_author(author):
    """Analyze author credibility"""
    score = 100
    issues = []
    
    if not author or author.lower() in ['unknown', 'anonymous', 'staff', 'admin']:
        score -= 30
        issues.append("No verified author provided")
    
    return max(0, score), issues

def analyze_date(date_str):
    """Check if date is recent and valid"""
    score = 100
    issues = []
    
    if not date_str:
        score -= 20
        issues.append("No publication date provided")
        return score, issues
    
    try:
        # Try to parse date (assuming ISO format or common formats)
        article_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        days_old = (datetime.now() - article_date).days
        
        if days_old > 365:
            score -= 10
            issues.append(f"Article is {days_old} days old")
    except:
        score -= 15
        issues.append("Invalid date format")
    
    return max(0, score), issues

def calculate_overall_credibility(scores):
    """Calculate weighted overall credibility score"""
    weights = {
        'title': 0.20,
        'content': 0.40,
        'author': 0.25,
        'date': 0.15
    }
    
    overall = sum(scores[key] * weights[key] for key in weights)
    
    if overall >= 80:
        verdict = "Likely Credible"
        confidence = "High"
    elif overall >= 60:
        verdict = "Possibly Credible"
        confidence = "Medium"
    elif overall >= 40:
        verdict = "Questionable"
        confidence = "Medium"
    else:
        verdict = "Likely Fake/Misleading"
        confidence = "High"
    
    return round(overall, 2), verdict, confidence

@app.route('/api/analyze', methods=['POST'])
def analyze_article():
    """Main endpoint for analyzing articles"""
    try:
        data = request.get_json()
        
        # Extract article components
        title = data.get('title', '')
        content = data.get('content', '')
        author = data.get('author', '')
        date = data.get('date', '')
        
        if not title or not content:
            return jsonify({
                'error': 'Title and content are required'
            }), 400
        
        # Analyze each component
        title_score, title_issues = analyze_title(title)
        content_score, content_issues = analyze_content(content)
        author_score, author_issues = analyze_author(author)
        date_score, date_issues = analyze_date(date)
        
        # Calculate overall credibility
        scores = {
            'title': title_score,
            'content': content_score,
            'author': author_score,
            'date': date_score
        }
        
        overall_score, verdict, confidence = calculate_overall_credibility(scores)
        
        # Compile all issues
        all_issues = {
            'title': title_issues,
            'content': content_issues,
            'author': author_issues,
            'date': date_issues
        }
        
        response = {
            'overall_score': overall_score,
            'verdict': verdict,
            'confidence': confidence,
            'component_scores': scores,
            'issues': all_issues,
            'recommendations': generate_recommendations(all_issues)
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_recommendations(issues):
    """Generate recommendations based on detected issues"""
    recommendations = []
    
    all_issue_list = []
    for category in issues.values():
        all_issue_list.extend(category)
    
    if len(all_issue_list) == 0:
        recommendations.append("Article appears credible, but always verify from multiple sources")
    elif len(all_issue_list) <= 2:
        recommendations.append("Minor concerns detected. Cross-reference with established news sources")
    else:
        recommendations.append("Multiple red flags detected. Verify this information independently")
        recommendations.append("Check if reputable news outlets are reporting the same story")
        recommendations.append("Look for primary sources and official statements")
    
    return recommendations

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()}), 200

@app.route('/api/batch', methods=['POST'])
def batch_analyze():
    """Batch analysis endpoint"""
    try:
        data = request.get_json()
        articles = data.get('articles', [])
        
        if not articles:
            return jsonify({'error': 'No articles provided'}), 400
        
        results = []
        for article in articles:
            title = article.get('title', '')
            content = article.get('content', '')
            author = article.get('author', '')
            date = article.get('date', '')
            
            if not title or not content:
                results.append({'error': 'Missing title or content'})
                continue
            
            title_score, title_issues = analyze_title(title)
            content_score, content_issues = analyze_content(content)
            author_score, author_issues = analyze_author(author)
            date_score, date_issues = analyze_date(date)
            
            scores = {
                'title': title_score,
                'content': content_score,
                'author': author_score,
                'date': date_score
            }
            
            overall_score, verdict, confidence = calculate_overall_credibility(scores)
            
            results.append({
                'title': title,
                'overall_score': overall_score,
                'verdict': verdict,
                'confidence': confidence
            })
        
        return jsonify({'results': results}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)