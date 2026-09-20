function export_phaseret_rtisila(outFile)
%EXPORT_PHASERET_RTISILA  PHASERET's RTISI-LA family on small Gabor frames.
%
%   export_phaseret_rtisila()
%   export_phaseret_rtisila(outFile)
%
%   Writes the reference that tests/regressions/test_rtisila_family.py
%   compares cool-frames' Gabor-domain rtisila, gsrtisila and lertisila
%   against (default outFile: tests/reference_data/phaseret_rtisila.mat).
%   Needs LTFAT and PHASERET on the path (ltfatstart; phaseretstart).  Runs
%   in Octave without PHASERET's MEX files, with three stand-ins for the
%   parts that exist only as C:
%
%   comp_leglaupdatesinglecol.m (this folder)
%       lertisila's column update, which PHASERET ships only as a MEX file:
%       a line-by-line transcription of libphaseret's
%       leglaupdate_col_execute with EXT_UPDOWN.
%   comp_spsi (written to a temporary folder)
%       PHASERET's comp/comp_spsireal.m, which defines comp_spsi under the
%       wrong file name, with the C library's two differences: log(s +
%       realmin) rather than log(s + eps), and a peak with p == 0 spreads
%       its phase nowhere (the MATLAB copy reuses the previous peak's
%       neighbours).
%   comp_gsrtisilaupdate / gsrtisila_c (written to a temporary folder)
%       PHASERET's MATLAB files with the C library's behaviour: the newest
%       frame's initial coefficients are synthesised into its time frame
%       before the update (the MATLAB fallback never reads them), and
%       'input' initialises the first lookahead frames (the MATLAB line that
%       means to has mismatched sizes).
%
%   Stored per case: the lattice, window, signal, the dgtreal coefficients
%   s ('timeinv'), and for every variant the coefficients c (PHASERET's
%   'timeinv' unless the name says freqinv) and, for lertisila, PHASERET's
%   relres (which lertisila computes correctly; rtisila's and gsrtisila's
%   is the norm of a complex difference, a phase measure).

if nargin < 1
  here = fileparts(mfilename('fullpath'));
  outFile = fullfile(here, '..', 'reference_data', 'phaseret_rtisila.mat');
end
here = fileparts(mfilename('fullpath'));
addpath(here);
tmp = tempname(); mkdir(tmp); addpath(tmp);
root = fileparts(which('phaseretstart'));

% comp_spsi with the C semantics
src = fileread(fullfile(root, 'comp', 'comp_spsireal.m'));
src = strrep(src, 'alpha=log(sabs(m-1)+eps);', ...
             sprintf('binup = m; bindown = m;\n            alpha=log(sabs(m-1)+realmin);'));
src = strrep(src, 'beta=log(sabs(m)+eps);', 'beta=log(sabs(m)+realmin);');
src = strrep(src, 'gamma=log(sabs(m+1)+eps);', 'gamma=log(sabs(m+1)+realmin);');
write_file(fullfile(tmp, 'comp_spsi.m'), src);

% gsrtisila with the C library's initialisation
src = fileread(fullfile(root, 'comp', 'comp_gsrtisilaupdate.m'));
old = sprintf('lookback = N - lookahead - 1;\n');
assert(numel(strfind(src, old)) == 1);
src = strrep(src, old, [old sprintf(['cframes(:,end) = gdnum.*(circshift(' ...
  'comp_ifftreal(coefbuf(:,end),M),floor(M/2)))*M;\n'])]);
write_file(fullfile(tmp, 'comp_gsrtisilaupdate.m'), src);
src = fileread(fullfile(root, 'gabor', 'gsrtisila.m'));
old = '    cframes(:,2:end) = s(:,1:kv.lookahead);';
assert(numel(strfind(src, old)) == 1);
src = strrep(src, old, sprintf(['    cframes(:,end-kv.lookahead+1:end) = s(:,1:kv.lookahead);\n' ...
  '    for ii = size(frames,2)-kv.lookahead+1:size(frames,2)\n' ...
  '        frames(:,ii) = fftshift(gabwin(gabdual(g,a,M,L),a,M)).*' ...
  '(circshift(comp_ifftreal(cframes(:,ii),M),floor(M/2)))*M;\n    end']));
src = strrep(src, 'function [c,f,relres,iter]=gsrtisila(', ...
             'function [c,f,relres,iter]=gsrtisila_c(');
write_file(fullfile(tmp, 'gsrtisila_c.m'), src);

% case A: gl = M, default look-ahead; case B: gl < M, lookahead 1
cases = {{32, 128, 128, 1024, []}, {24, 96, 72, 576, 1}};
R = struct();
for ci = 1:numel(cases)
  cc = cases{ci}; a = cc{1}; M = cc{2}; gl = cc{3}; L = cc{4}; la = cc{5};
  p = sprintf('c%d_', ci);
  t = (0:L-1)'/8000;
  f = sin(2*pi*(300*t + 900*t.^2)) + 0.5*sin(2*pi*1234.5*t).*exp(-3*t) ...
      + 0.1*cos(2*pi*37*t.^1.5);
  g = firwin('hann', gl);
  s = dgtreal(f, g, a, M, 'timeinv');
  abss = abs(s);
  R.([p 'a']) = a; R.([p 'M']) = M; R.([p 'g']) = g; R.([p 'f']) = f; R.([p 's']) = s;
  if isempty(la); args = {}; else; args = {'lookahead', la}; end
  R.([p 'lookahead']) = la;
  R.([p 'rt_c']) = rtisila(abss, g, a, M, args{:});
  R.([p 'rt3fi_c']) = rtisila(abss, g, a, M, args{:}, 'freqinv', 'maxit', 3);
  inits = {'zeros', 'unwrap', 'spsi', 'input'};
  for k = 1:numel(inits)
    if strcmp(inits{k}, 'input'); sin_ = s; else; sin_ = abss; end
    R.([p 'gs_' inits{k} '_c']) = gsrtisila_c(sin_, g, a, M, args{:}, inits{k});
  end
  if isempty(la)
    variants = {{'zhu'}, {'zero', 'regwin'}, {'zhu', 'modtrunc'}, ...
                {'input', 'energy'}, {'zhu', 'onthefly'}, {'unwrap'}};
    for k = 1:numel(variants)
      v = variants{k};
      if strcmp(v{1}, 'input'); sin_ = s; else; sin_ = abss; end
      [c, ~, relres] = lertisila(sin_, g, a, M, v{:});
      q = sprintf('%sle%d_', p, k);
      R.([q 'c']) = c; R.([q 'relres']) = relres;
    end
  end
end
save('-v7', outFile, '-struct', 'R');
rmpath(tmp);
end

function write_file(name, txt)
fid = fopen(name, 'w'); fputs(fid, txt); fclose(fid);
end
