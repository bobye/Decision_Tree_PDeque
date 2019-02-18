#include "core/decision_tree.hpp"

namespace d2 {

  template <size_t dim, class YStats>
  std::ostream& operator<<(std::ostream &f, const internal::_DTNode<dim, YStats>& node) {
    return f;
  };  

  template <size_t dim, class YStats>
  std::ostream& operator<<(std::ostream &f, const internal::_DTLeaf<dim, YStats>& leaf) {
    f << "node" << leaf.hashCode() << " [label=\"" << leaf.label << "\", shape=box, style=filled ]" << std::endl;
    return f;
  } 

  template <size_t dim, class YStats>
  std::ostream& operator<<(std::ostream &f, const internal::_DTBranch<dim, YStats>& branch) {
    assert(branch.left && branch.right);
    f << *(branch.left);
    f << *(branch.right);    
    f << "node" << branch.hashCode() << " [label=\"x" << branch.index << " < " << branch.cutoff << "?\", style=filled]" << std::endl;
    f << "node" << branch.hashCode() << " -> node" << branch.left->hashCode() << " [label=\"yes\"]" << std::endl;
    f << "node" << branch.hashCode() << " -> node" << branch.right->hashCode() << "[label=\"no\"]" << std::endl;
    return f;
  }

  template <size_t dim, class YStats, typename criterion>
  std::ostream& operator<<(std::ostream &f, const Decision_Tree<dim, YStats, criterion>& dt) {
    assert(dt.root);
    f << "digraph G {\n";
    f << *(dt.root);
    f << "}\n";      
    return f;
  }
}
