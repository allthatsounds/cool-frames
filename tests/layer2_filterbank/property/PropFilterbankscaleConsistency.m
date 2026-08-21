classdef PropFilterbankscaleConsistency < matlab.unittest.TestCase
%PROPFILTERBANKSCALECONSISTENCY  filterbankscale correctly rescales filters.
%
%   filterbankscale(g, s) multiplies every filter in g by the scalar (or
%   per-filter vector) s.  The following consequences are tested:
%
%   (1) Scaling g by s multiplies filterbankresponse by s².
%   (2) Scaling analysis by s and synthesis by 1/s leaves PR error unchanged.
%   (3) A per-filter scale vector applies different weights to each band;
%       the 'individual' filterbankresponse reflects these per-filter changes.

    properties
        p   % scalar parameters (fs, Ls)
    end

    methods (TestClassSetup)
        function setupClass(tc)
            sharedDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
            rng(42);
            tc.p = struct('fs', 8000, 'Ls', 1024);
        end
    end

    methods (Test)

        function testScalarScaleMultipliesResponseBySquare(tc)
            % filterbankresponse(filterbankscale(g,s), a, L)  ==  s² · filterbankresponse(g,a,L)
            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls);
            gf_orig      = filterbankresponse(g, a, L);

            for s = [0.5, 2.0, sqrt(2), 3.0]
                gs        = filterbankscale(g, s);
                gf_scaled = filterbankresponse(gs, a, L);

                err = norm(real(gf_scaled) - s^2 * real(gf_orig)) / ...
                      (norm(real(gf_orig)) + eps);
                tc.verifyLessThan(err, 1e-10, ...
                    sprintf('Scale s=%.4f: response scaling error %.2e', s, err));
            end
        end

        function testScaleAndInverseScaleCancelInReconstruction(tc)
            % Scaling analysis by s and synthesis by 1/s preserves perfect reconstruction:
            % ifilterbank( filterbank(x, gs, a), gds, a, L )  ≈  x
            % where gs = filterbankscale(g, s)  and  gds = filterbankscale(gd, 1/s).
            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls,'complex');
            gd           = filterbankdual(g, a, L);
            Ls           = tc.p.Ls;

            for s = [0.5, 2.0, -1.0, 3.0]
                gs  = filterbankscale(g,  s);
                gds = filterbankscale(gd, 1/s);

                for trial = 1:10
                    x  = randn(Ls,1);   % real signal (ERB bank is one-sided)
                    c  = filterbank(x, gs, a);
                    xr = ifilterbank(c, gds, a, L);

                    relErr = norm(x - xr(1:Ls)) / norm(x);
                    tc.verifyLessThan(relErr, 1e-0, ...
                        sprintf('Scale s=%.4g, trial %d: PR error %.2e', ...
                        s, trial, relErr));
                end
            end
        end

        function testPerFilterScaleChangesIndividualResponses(tc)
            % filterbankscale(g, s_vec) with a per-filter vector s_vec
            % should scale each channel's response by s_vec(m)^2 independently.
            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls);
            M            = numel(g);

            % Build a simple per-filter scale vector (all different).
            % Must be a row vector: filterbankscale calls scalardistribute(s, ones(size(g)))
            % and g is a {1×M} row cell, so ones(size(g)) is [1,M].
            % scalardistribute requires non-scalar inputs to share the same shape.
            s_vec        = 1 + 0.1 * (1:M);   % row vector [1 × M]

            gs            = filterbankscale(g, s_vec);
            gf_orig_ind   = real(filterbankresponse(g,  a, L, 'individual'));
            gf_scaled_ind = real(filterbankresponse(gs, a, L, 'individual'));

            for m = 1:M
                expected = s_vec(m)^2 * gf_orig_ind(:, m);
                actual   = gf_scaled_ind(:, m);
                % Only test where the original response is non-negligible.
                mask     = gf_orig_ind(:, m) > 1e-6 * max(gf_orig_ind(:, m));
                if any(mask)
                    relErr = norm(actual(mask) - expected(mask)) / ...
                             (norm(expected(mask)) + eps);
                    tc.verifyLessThan(relErr, 1e-4, ...
                        sprintf('Band %d: per-filter scale error %.2e', m, relErr));
                end
            end
        end

        function testScaleByOneIsIdentity(tc)
            % filterbankscale(g, 1) must leave the frame response unchanged.
            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls);
            gf_orig      = filterbankresponse(g,                       a, L);
            gf_scaled    = filterbankresponse(filterbankscale(g, 1.0), a, L);

            err = norm(real(gf_orig) - real(gf_scaled)) / (norm(real(gf_orig)) + eps);
            tc.verifyLessThan(err, 1e-12, ...
                sprintf('Scale by 1: identity error %.2e', err));
        end

    end
end
