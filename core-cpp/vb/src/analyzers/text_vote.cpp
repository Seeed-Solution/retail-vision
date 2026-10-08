#include "text_vote.h"
#include "vb/json.h"
#include "vb/text_match.h"
#include <algorithm>
#include <deque>
#include <map>
#include <set>
#include <cmath>

namespace vb { namespace {
struct VoteCfg { int window=7,min_agree=3,flush=0; float min_conf=0; double dedup=0; std::vector<TextPattern> accept; };
struct Vote { Stage2Read r; };
class TextVote final : public Analyzer {
 public:
  const char* name() const override { return "text_vote"; }
  void set_stage2_available(bool v) override { stage2_=v; }
  uint64_t text_vote_dedup_count() const override { return dedup_count_; }
  bool configure(const std::string& s,std::string& e) override {
    Json j; try{j=Json::parse(s);}catch(...){e="text_vote: invalid config";return false;}
    if(!j.is_object()){e="text_vote: config must be an object";return false;}
    VoteCfg n=cfg_; n.accept.clear(); auto sit=j.find("source"); if(sit==j.end()||!sit->is_string()||sit->get<std::string>()!="stage2"){e="text_vote source must be stage2";return false;} if(!stage2_){e="text_vote needs stage2";return false;}
    auto integer = [&](const char* k, int def, int& dst) { auto it=j.find(k); if(it==j.end()){dst=def;return true;} if(!it->is_number_integer()){e=std::string("text_vote: ")+k+" must be integer";return false;} constexpr int64_t lo=std::numeric_limits<int>::min(), hi=std::numeric_limits<int>::max(); if(it->is_number_unsigned()){auto v=it->get<uint64_t>();if(v>static_cast<uint64_t>(hi)){e=std::string("text_vote: ")+k+" out of range";return false;}dst=static_cast<int>(v);}else{auto v=it->get<int64_t>();if(v<lo||v>hi){e=std::string("text_vote: ")+k+" out of range";return false;}dst=static_cast<int>(v);} return true; };
    auto number = [&](const char* k, double def, double& dst) { auto it=j.find(k); if(it==j.end()){dst=def;return true;} if(!it->is_number()){e=std::string("text_vote: ")+k+" must be number";return false;} dst=it->get<double>(); if(!std::isfinite(dst)){e=std::string("text_vote: ")+k+" out of range";return false;} return true; };
    int window, min_agree, flush; double min_conf, dedup;
    if(!integer("window",7,window)||!integer("min_agree",3,min_agree)||!integer("flush_min_agree",0,flush)||!number("min_conf",0,min_conf)||!number("dedup_s",0,dedup)) return false;
    if(window<1||window>64||min_agree<1||min_agree>window||flush<0||flush>min_agree||min_conf<0||min_conf>1||dedup<0){e="text_vote: thresholds invalid";return false;}
    n.window=window;n.min_agree=min_agree;n.flush=flush;n.min_conf=static_cast<float>(min_conf);n.dedup=dedup;
    auto a=j.find("accept"); if(a!=j.end()&&!parse_text_patterns(*a,n.accept,e))return false;
    cfg_=std::move(n); return true;
  }
  void on_frame(const FrameMeta&m,const std::vector<Track>&,float*,std::vector<AnalyzerEvent>&out) override { if(!m.reads)return; for(const auto&r:*m.reads){bool ok=true;for(float x:r.bbox)ok=ok&&std::isfinite(x)&&x>=0&&x<=1;if(!ok||r.bbox[0]>=r.bbox[2]||r.bbox[1]>=r.bbox[3]||!std::isfinite(r.mean_conf)||!std::isfinite(r.min_char_conf)||r.mean_conf<0||r.mean_conf>1||r.min_char_conf<0||r.min_char_conf>1)continue;if(emitted_.count(r.track_id)||r.text.empty()||r.min_char_conf<cfg_.min_conf||!text_matches(r.text,cfg_.accept))continue;auto&q=buf_[r.track_id];q.push_back({r});while((int)q.size()>cfg_.window)q.pop_front();std::map<std::string,int> counts;std::string top;int n=0;for(auto&v:q){int x=++counts[v.r.text];if(x>n){n=x;top=v.r.text;}}if(n>=cfg_.min_agree)emit(r.track_id,top,false,m.t_mono_s,out);}}
  void on_track_removed(uint32_t id,double t,std::vector<AnalyzerEvent>&out) override {auto it=buf_.find(id);if(it!=buf_.end()&&!emitted_.count(id)&&cfg_.flush>0){std::map<std::string,int> c;std::string top;int n=0;for(auto&v:it->second){int x=++c[v.r.text];if(x>n){n=x;top=v.r.text;}}if(n>=cfg_.flush)emit(id,top,true,t,out);}buf_.erase(id);emitted_.erase(id);}
 private:
  void emit(uint32_t id,const std::string&text,bool flushed,double now,std::vector<AnalyzerEvent>&out){if(emitted_.count(id))return;emitted_.insert(id);auto&q=buf_[id];std::vector<const Stage2Read*> a;for(auto&v:q)if(v.r.text==text)a.push_back(&v.r);if(a.empty())return;const Stage2Read*best=a[0];float sum=0;for(auto*x:a){sum+=x->mean_conf;if(x->min_char_conf>best->min_char_conf||(x->min_char_conf==best->min_char_conf&&x->seq<best->seq))best=x;}auto it=last_.find(text);if(cfg_.dedup>0&&it!=last_.end()&&now-it->second<cfg_.dedup){++dedup_count_;return;}last_[text]=now;Json f;f["text"]=text;f["votes"]=a.size();f["frames"]=q.size();f["conf"]=sum/a.size();f["min_char_conf"]=best->min_char_conf;f["bbox"]={best->bbox[0],best->bbox[1],best->bbox[2],best->bbox[3]};f["best_seq"]=best->seq;f["flushed"]=flushed;out.push_back({"text_read",id,json_dump(f)});}
  VoteCfg cfg_; bool stage2_=false;std::map<uint32_t,std::deque<Vote>>buf_;std::set<uint32_t>emitted_;std::map<std::string,double>last_;uint64_t dedup_count_=0;
}; }
std::unique_ptr<Analyzer> make_text_vote_analyzer(){return std::make_unique<TextVote>();}
}
