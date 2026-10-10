'use strict';
(() => {
  const labels = {has_five:'已有五连', black_count:'黑子计数', white_count:'白子计数',
    can_win_now:'己方一步获胜', opponent_has_immediate_win:'对方一步威胁',
    winning_move:'选择获胜落点', blocking_move:'选择阻挡落点'};
  const examples = window.GOMOKU_EXAMPLES || [];
  const $ = id => document.getElementById('example-' + id);
  const text = (id, value) => $(id).textContent = value;
  const display = v => typeof v === 'boolean' ? (v ? '是' : '否') : v === 'abstain' ? '弃答（以上皆非）' : String(v);
  const pct = p => (p*100).toFixed(2) + '%';
  let visible = [], index = 0;

  function question(row) {
    const color = row.side_to_move === 'black' ? '黑方' : '白方';
    return {has_five:'当前棋盘上，任一方是否已有连续至少五子？',
      black_count:'棋盘上有多少颗黑子？', white_count:'棋盘上有多少颗白子？',
      can_win_now:`轮到${color}落子，下一手是否能形成至少五连？`,
      opponent_has_immediate_win:`当前轮到${color}。假设对方落一子，能否形成至少五连？`,
      winning_move:`轮到${color}落子。四个候选空位中，哪个能立即形成至少五连？`,
      blocking_move:`轮到${color}落子。四个候选空位中，哪个能消除对方全部一步五连威胁？`}[row.qid];
  }
  function resultMatch(row, filter) {
    const q = row.qwen.correct, o = row.omnijev.correct;
    return {different:row.qwen.prediction !== row.omnijev.prediction,
      'omni-only':o && !q, 'qwen-only':q && !o,
      'both-right':q && o, 'both-wrong':!q && !o, all:true}[filter];
  }
  function refresh() {
    const task = $('task').value, filter = $('result').value;
    const query = $('search').value.trim().toLowerCase(), current = visible[index]?.id;
    visible = examples.filter(row => (task === 'all' || row.qid === task) && resultMatch(row, filter)
      && (!query || row.id.toLowerCase().includes(query)));
    const keep = visible.findIndex(x => x.id === current);
    index = keep < 0 ? 0 : keep;
    text('count', visible.length + ' / ' + examples.length + ' 道题');
    $('select').replaceChildren(...visible.map((row, i) => {
      const option = document.createElement('option');
      option.value = String(i); option.textContent = row.id + ' · ' + labels[row.qid];
      return option;
    }));
    $('empty').hidden = visible.length > 0;
    $('layout').hidden = $('raw').hidden = !visible.length;
    draw();
  }
  function renderOptions(row, name, prefix) {
    const model = row[name], container = $(prefix + '-options');
    container.replaceChildren();
    const options = row.type === 'noul'
      ? [{key:true, label:'是', p:model.probability_yes}, {key:false, label:'否', p:1-model.probability_yes}]
      : [...row.options.map(o => ({key:o.key, label:o.text, p:model.probabilities[o.key]})),
         {key:'abstain', label:'弃答（以上皆非）', p:model.abstain}];
    for (const option of options) {
      const picked = option.key === model.prediction, gold = option.key === row.gold;
      const item = document.createElement('div');
      item.className = 'answer-option' + (picked ? ' picked' : '') + (gold ? ' gold-option' : '');
      const top = document.createElement('div'); top.className = 'option-top';
      const label = document.createElement('span'); label.textContent = option.label;
      const value = document.createElement('b'); value.textContent = pct(option.p);
      top.append(label, value);
      const tags = document.createElement('small');
      tags.textContent = [picked ? '模型选择' : '', gold ? '正确答案' : ''].filter(Boolean).join(' · ');
      const bar = document.createElement('div'); bar.className = 'option-bar';
      const fill = document.createElement('i'); fill.style.width = Math.max(0, Math.min(100, option.p*100)) + '%';
      bar.append(fill); item.append(top, tags, bar); container.append(item);
    }
  }
  function draw() {
    const empty = !visible.length;
    $('prev').disabled = empty || index === 0;
    $('next').disabled = empty || index === visible.length - 1;
    $('select').disabled = empty;
    text('position', empty ? '0 / 0' : (index+1) + ' / ' + visible.length);
    if (empty) return;
    const row = visible[index];
    $('select').value = String(index);
    $('image').src = row.image; $('original').href = row.image; $('image').alt = row.record_id + ' 五子棋原始输入图片';
    text('caption', '棋盘 ' + row.record_id + ' · 列 A–O，行 1–15');
    text('task-name', labels[row.qid]); text('id', row.id);
    text('question', question(row)); text('prompt', '原始评测题目：' + row.prompt);
    text('gold', display(row.gold));
    for (const [name, prefix] of [['qwen','qwen'], ['omnijev','omni']]) {
      const model = row[name];
      text(prefix + '-status', model.correct ? '答对' : '答错');
      $(prefix + '-status').className = model.correct ? 'answer-good' : 'answer-bad';
      text(prefix + '-answer', display(model.prediction));
      text(prefix + '-confidence', row.type === 'noul' ? 'P(是) ' + pct(model.probability_yes) : '弃答 ' + pct(model.abstain));
      renderOptions(row, name, prefix);
    }
    text('json', JSON.stringify(row, null, 2));
  }
  function move(delta) {
    if (!visible.length) return;
    index = Math.max(0, Math.min(visible.length-1, index + delta));
    draw();
  }
  $('task').onchange = $('result').onchange = refresh;
  $('search').oninput = refresh;
  $('select').onchange = () => {index = Number($('select').value); draw();};
  $('prev').onclick = () => move(-1); $('next').onclick = () => move(1);
  document.addEventListener('keydown', event => {
    if (['INPUT','SELECT','TEXTAREA','BUTTON','SUMMARY'].includes(event.target.tagName)
      || event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      event.preventDefault(); move(event.key === 'ArrowLeft' ? -1 : 1);
    }
  });
  refresh();
})();
