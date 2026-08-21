classdef PropBiquadFilterbank < matlab.unittest.TestCase
%PROPBIQUADFILTERBANK  Property tests for biquadfilter in the filterbank pipeline.
%
%   Verifies that biquadfilter-based filterbanks respect the same frame-
%   theoretic properties as blfilter-based ones:
%
%   (1) filterbankresponse returns a real, positive-valued vector.
%   (2) filterbankbounds returns finite, ordered 0 < A <= B.
%   (3) filterbank analysis is linear.
%   (4) Weighted coefficient energy satisfies the frame inequality
%       A ||x||^2 <= sum_m (1/a_m) ||c_m||^2 <= B ||x||^2 for complex x.
%   (5) filterbankscale multiplies the frame response by s^2.
%   (6) filterbankdual + ifilterbank gives perfect reconstruction for
%       complex-valued signals analysed with a full-bandwidth biquad bank.

    properties
        p       % parameters: fs, Ls
        g       % cell array of biquad analysis filters
        a       % hop-size vector (all-ones: no subsampling)
        L       % transform length
        M       % number of biquad channels
    end

    methods (TestClassSetup)
        function setupClass(tc)
            sharedDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
            rng(42);

            tc.p = struct('fs', 8000, 'Ls', 512);

            % Build a small biquad filterbank with 6 resonators spread across
            % normalized frequencies 0.1 … 0.9 so that the bank covers the
            % full complex spectrum (no one-sided truncation).
            M_biq  = 6;
            fcs    = linspace(0.1, 0.9, M_biq);   % normalised, LTFAT: 1 = Nyquist
            bws    = 0.12 * ones(1, M_biq);
            tc.g   = biquadfilter(fcs, bws);       % returns cell array

            % No subsampling (a=1 ensures painless condition is satisfied
            % trivially and frame bounds can be computed reliably).
            tc.a   = ones(M_biq, 1);
            tc.L   = filterbanklength(tc.p.Ls, tc.a);
            tc.M   = M_biq;
        end
    end

    methods (Test)

        function testBiquadResponseIsRealAndPositive(tc)
            % filterbankresponse must be real and non-negative everywhere.
            gf = real(filterbankresponse(tc.g, tc.a, tc.L));
            tc.verifyTrue(all(gf(:) >= -1e-12), ...
                'biquadfilter: filterbankresponse must be non-negative.');
        end

        function testBiquadFrameBoundsOrderedAndFinite(tc)
            [A, B] = filterbankbounds(tc.g, tc.a, tc.L);
            tc.verifyGreaterThan(A, 0, ...
                'biquadfilter bank: lower frame bound A must be positive.');
            tc.verifyLessThan(B, Inf, ...
                'biquadfilter bank: upper frame bound B must be finite.');
            tc.verifyLessThanOrEqual(A, B + 1e-12, ...
                'biquadfilter bank: frame bounds must satisfy A <= B.');
        end

        function testBiquadAnalysisIsLinear(tc)
            % T(alpha*x + beta*y) = alpha*T(x) + beta*T(y)  for complex signals.
            alpha = 1.5 + 0.7i;
            beta  = -0.3 + 1.2i;
            Ls    = tc.p.Ls;
            rng(7);
            x = randn(Ls, 1) + 1i*randn(Ls, 1);
            y = randn(Ls, 1) + 1i*randn(Ls, 1);

            cx     = filterbank(x,               tc.g, tc.a);
            cy     = filterbank(y,               tc.g, tc.a);
            csum   = filterbank(alpha*x + beta*y, tc.g, tc.a);

            for m = 1 : tc.M
                expected = alpha * cx{m} + beta * cy{m};
                relErr   = norm(csum{m} - expected) / (norm(expected) + eps);
                tc.verifyLessThan(relErr, 1e-10, ...
                    sprintf('biquadfilter linearity violated in channel %d.', m));
            end
        end

        function testBiquadFrameInequalityHolds(tc)
            % A ||x||^2 <= sum_m (1/a_m)||c_m||^2 <= B ||x||^2  for 20 signals.
            [A, B] = filterbankbounds(tc.g, tc.a, tc.L);
            Ls     = tc.p.Ls;
            rng(42);
            for trial = 1 : 20
                x   = randn(Ls, 1) + 1i*randn(Ls, 1);
                c   = filterbank(x, tc.g, tc.a);
                ex  = norm(x)^2;
                eTx = sum(cellfun(@(cm, am) norm(cm)^2 / am, c, num2cell(tc.a)));
                tc.verifyGreaterThanOrEqual(eTx, (A - 1e-6) * ex, ...
                    sprintf('Trial %d: weighted energy below lower bound A.', trial));
                tc.verifyLessThanOrEqual(eTx, (B + 1e-6) * ex, ...
                    sprintf('Trial %d: weighted energy above upper bound B.', trial));
            end
        end

        function testBiquadScaleMultipliesResponseBySquare(tc)
            % filterbankscale(g, s): response should scale by s^2.
            gf_orig = real(filterbankresponse(tc.g, tc.a, tc.L));
            for s = [0.5, 2.0, 3.0]
                gs        = filterbankscale(tc.g, s);
                gf_scaled = real(filterbankresponse(gs, tc.a, tc.L));
                relErr    = norm(gf_scaled - s^2 * gf_orig) / (norm(gf_orig) + eps);
                tc.verifyLessThan(relErr, 1e-10, ...
                    sprintf('biquadfilter scale s=%.1f: response scaling error.', s));
            end
        end

        function testBiquadDualFramePerfectReconstruction(tc)
            % For the complex-valued biquad bank with a=1, filterbankdual must
            % give perfect reconstruction for complex signals.
            gd  = filterbankdual(tc.g, tc.a, tc.L);
            Ls  = tc.p.Ls;
            rng(17);
            for trial = 1 : 5
                x  = randn(Ls, 1) + 1i*randn(Ls, 1);
                c  = filterbank(x, tc.g, tc.a);
                xr = ifilterbank(c, gd, tc.a, tc.L);
                relErr = norm(x - xr(1:Ls)) / norm(x);
                tc.verifyLessThan(relErr, 1e-6, ...
                    sprintf('biquadfilter dual PR trial %d: error %.2e.', trial, relErr));
            end
        end

        function testBiquadTightFrameEqualBounds(tc)
            % filterbanktight must produce equal positive-frequency bounds.
            gt = filterbanktight(tc.g, tc.a, tc.L);
            [A_t, B_t] = filterbankbounds(gt, tc.a, tc.L);
            rel_diff = abs(B_t - A_t) / (A_t + eps);
            tc.verifyLessThan(rel_diff, 1e-4, ...
                'biquadfilter tight frame: bounds A and B must be approximately equal.');
        end

        function testBiquadResponsenPerChannelConsistency(tc)
            % The individual per-channel response from filterbankresponse must
            % sum to the total response.
            gf_total = real(filterbankresponse(tc.g, tc.a, tc.L));
            gf_indiv = real(filterbankresponse(tc.g, tc.a, tc.L, 'individual'));
            gf_sum   = sum(gf_indiv, 2);
            relErr   = norm(gf_total - gf_sum) / (norm(gf_total) + eps);
            tc.verifyLessThan(relErr, 1e-10, ...
                'biquadfilter: sum of individual responses must equal total response.');
        end

    end
end
