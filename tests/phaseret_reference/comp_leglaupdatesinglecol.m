function cout = comp_leglaupdatesinglecol(c, kern, s, M, do_onthefly)
% Line-by-line transcription of PHASERET's mex/comp_leglaupdatesinglecol.c
% (libphaseret leglaupdate_col_execute with EXT_UPDOWN), for validation only.
M2 = floor(M/2) + 1;
kernh2 = size(kern,1); kernw = size(kern,2); kernh = 2*kernh2 - 1;
M2buf = M2 + kernh - 1;
kernwMidId = floor(kernw/2);        % 0-based
buf = zeros(M2buf, kernw);
buf(kernh2:kernh2+M2-1, :) = c;     % rows kernh2-1 .. (0-based)
for n = 1:kernw
  for m = 0:kernh2-2                % top: e[kernh2-2-m] = conj(c[1+m])
    buf(kernh2-2-m + 1, n) = conj(c(1+m + 1, n));
  end
  for m = 0:kernh2-2                % bottom: e[kernh2-1+M2+m] = conj(c[M2-2+M%2-m])
    buf(kernh2-1+M2+m + 1, n) = conj(c(M2-2+mod(M,2)-m + 1, n));
  end
end
cout = zeros(M2,1);
for mfirst = 0:M2-1
  m = mfirst + kernh2 - 1; mlast = mfirst + kernh - 1;
  accum = 0;
  for kn = 0:kernw-1
    for km = 0:kernh2-2
      a = kern(kernh2-1-km + 1, kn+1);
      accum = accum + a*buf(mfirst+km + 1, kn+1) + conj(a)*buf(mlast-km + 1, kn+1);
    end
    accum = accum + kern(1, kn+1)*buf(m + 1, kn+1);
  end
  cout(mfirst+1) = accum;
  if do_onthefly
    cout(mfirst+1) = s(mfirst+1)*exp(1i*angle(cout(mfirst+1)));
    buf(m + 1, kernwMidId + 1) = cout(mfirst+1);
  end
end
if ~do_onthefly
  cout = s.*exp(1i*angle(cout));
end
