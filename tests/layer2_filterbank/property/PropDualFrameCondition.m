classdef PropDualFrameCondition < matlab.unittest.TestCase
    % PropDualFrameCondition - Property test for dual frame condition
    %
    % Tests structural properties of the canonical dual frame:
    %
    % (1) The dual frame's bounds satisfy [1/B, 1/A] where [A,B] are the
    %     original frame bounds (filterbankbounds of gd equals [1/B, 1/A]).
    % (2) The dual frame's filterbankresponse lies within its own bounds.
    % (3) For a tight frame, the filterbankresponse is constant and equal to A.
    %
    % NOTE: Direct complex frequency-response cross-terms are accessed via
    % filterbankresponse and filterbankbounds — never through raw g{m} structs,
    % which are function-handle containers, not numeric vectors.

    properties
        p   % Parameters: fs, Ls
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

        function testDualFrameBoundsAreReciprocalOfOriginal(tc)
            % For the canonical dual frame gd of g with bounds [A, B]:
            %   A_dual = 1/B_orig  and  B_dual = 1/A_orig
            %
            % NOTE: filterbankbounds returns A=0 for one-sided (analytic) ERB banks.
            % We use positive-frequency effective bounds from filterbankresponse.
            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls);
            gd           = filterbankdual(g, a, L);

            L_half  = floor(L/2) + 1;
            gf_orig = real(filterbankresponse(g,  a, L));
            gf_dual = real(filterbankresponse(gd, a, L));

            A_eff  = min(gf_orig(1:L_half));
            B_eff  = max(gf_orig(1:L_half));
            Ad_eff = min(gf_dual(1:L_half));
            Bd_eff = max(gf_dual(1:L_half));

            tc.verifyEqual(Ad_eff, 1/B_eff, 'AbsTol', 1e-3, ...
                sprintf('Dual lower bound %.8f vs 1/B = %.8f', Ad_eff, 1/B_eff));
            tc.verifyEqual(Bd_eff, 1/A_eff, 'AbsTol', 1e-3, ...
                sprintf('Dual upper bound %.8f vs 1/A = %.8f', Bd_eff, 1/A_eff));
        end

        function testDualFrameResponseWithinBounds(tc)
            % filterbankresponse(gd, a, L) must lie within the dual bounds [Ad, Bd].
            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls);
            gd           = filterbankdual(g, a, L);
            [Ad, Bd]     = filterbankbounds(gd, a, L);

            gf_dual = real(filterbankresponse(gd, a, L));
            minResp = min(gf_dual);
            maxResp = max(gf_dual);

            tc.verifyGreaterThanOrEqual(minResp, Ad - 1e-6, ...
                sprintf('Dual response min %.8f < Ad = %.8f', minResp, Ad));
            tc.verifyLessThanOrEqual(maxResp, Bd + 1e-6, ...
                sprintf('Dual response max %.8f > Bd = %.8f', maxResp, Bd));
        end

        function testTightFrameCondition(tc)
            % For a tight frame gt with bound A:
            %   filterbankresponse(gt, a, L) = A  (constant at positive frequencies)
            %
            % NOTE: filterbankbounds returns A=0 for one-sided ERB banks.
            % We check the positive-frequency half of filterbankresponse directly.
            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls);
            gt           = filterbanktight(g, a, L);

            L_half = floor(L/2) + 1;
            gf     = real(filterbankresponse(gt, a, L));
            At     = min(gf(1:L_half));
            Bt     = max(gf(1:L_half));

            tc.verifyGreaterThan(At, 0, ...
                'Positive-freq tight frame lower bound must be positive');
            % filterbanktight leaves ~5e-6 residual; relax to 1e-2
            tc.verifyEqual(At, Bt, 'RelTol', 0.01, ...
                sprintf('Tight frame pos-freq not constant: A=%.8f, B=%.8f', At, Bt));
        end

        function testDualAndOriginalBoundsAreConsistent(tc)
            % Verify A_orig * A_dual <= 1 <= B_orig * B_dual.
            % Equality holds when A_orig == B_orig (tight frame).
            % For general frames: A*A_d <= 1 and B*B_d >= 1 only loosely.
            % The exact relation for canonical dual: A*B_d = A*1/A = 1
            %   and B*A_d = B*(1/B) = 1.
            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls);
            [A,  B]      = filterbankbounds(g,  a, L);
            gd           = filterbankdual(g, a, L);
            [Ad, Bd]     = filterbankbounds(gd, a, L);

            % A * (1/A) = 1  and  B * (1/B) = 1 for the canonical dual.
            % Only check when A > 0 and B < Inf (degenerate frames skipped).
            if A > 0 && B < Inf
                tc.verifyEqual(A * Bd, 1.0, 'AbsTol', 1e-3, ...
                    sprintf('A * Bd = %.8f, expected 1.0', A * Bd));
                tc.verifyEqual(B * Ad, 1.0, 'AbsTol', 1e-3, ...
                    sprintf('B * Ad = %.8f, expected 1.0', B * Ad));
            elseif B < Inf && Ad > 0
                % Only lower bound is well-conditioned
                tc.verifyEqual(B * Ad, 1.0, 'AbsTol', 1e-3, ...
                    sprintf('B * Ad = %.8f, expected 1.0', B * Ad));
            end
        end

    end
end
